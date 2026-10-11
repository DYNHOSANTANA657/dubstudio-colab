'use strict';
// DOM-fake regressions execute production code without starting a browser/app.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const cases = [];
const a = {id:'aaa111aaa111',status:'awaiting_colab',has_colab_package:true,options:{operation:'dub_video',target_languages:['es-ES'],require_lipsync:true}};
const b = {...a,id:'bbb222bbb222'};
function payload(text) {return {job_id:a.id,translations:{'es-ES':[{id:1,start:0,end:1,speaker:'SPEAKER_00',text:'Source A',translated_text:text}]}};}
function deferred() {let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};}
function fixture({productionJobs=false,storage}={}) {
  let detail=null;
  const container={querySelector:selector=>selector==='[data-current-job]'?detail:null,querySelectorAll:()=>[]};
  const elements={'#jobDetail':container,'#jobList':{dataset:{},innerHTML:'',querySelectorAll:()=>[]},'#jobCount':{textContent:''},'#lastRefresh':{textContent:''}};
  const document={querySelector:selector=>elements[selector]||null,querySelectorAll:()=>[]};
  const context=vm.createContext({document,console,FormData,structuredClone,URL,...(storage?{sessionStorage:storage}:{})});
  vm.runInContext(source.replace(/init\(\);\s*$/, ''),context);
  const state=vm.runInContext('state',context);
  const productionApi=vm.runInContext('api',context);
  state.jobs=[a,b];
  const calls=[],toasts=[],downloads=[];
  context.api=async (url,options)=>{calls.push({url,options});return {};};
  context.toast=message=>toasts.push(message);
  context.downloadUrl=url=>downloads.push(url);
  if(!productionJobs) context.loadJobs=async()=>{};
  function select(job) {
    state.selectedJob=job.id;
    const editor={innerHTML:'',inputs:[],querySelectorAll:()=>editor.inputs};
    detail={dataset:{currentJob:job.id},editor,imports:[],querySelector:selector=>selector==='#reviewEditor'?editor:null,querySelectorAll:selector=>selector==='[data-import]'?detail.imports:[]};
    return editor;
  }
  function input(job,mode='result',explicit=true) {
    return {files:[new Blob(['synthetic zip'])],dataset:{import:mode,...(explicit?{importJob:job.id}:{})},disabled:false,value:'fixture.zip',closest:()=>({dataset:{currentJob:job.id}})};
  }
  return {context,state,productionApi,calls,toasts,downloads,container,document,elements,select,input,get detail(){return detail;},set detail(value){detail=value;}};
}
function libraryFixture(options={}) {
  const f=fixture({productionJobs:true,...options}),rendered=[];
  f.context.renderJobDetail=job=>{rendered.push(job.id);f.select(job);};
  return {...f,rendered};
}
function queuedLibrary(f) {
  const rounds=[];
  f.context.api=(url,options)=>{
    f.calls.push({url,options});
    if(url==='/api/jobs') {const round={jobs:deferred(),trash:deferred()};rounds.push(round);return round.jobs.promise;}
    if(url==='/api/trash') return rounds.at(-1).trash.promise;
    throw new Error(`Unexpected fixture request: ${url}`);
  };
  return {rounds,complete(index,jobs,trash=[]){rounds[index].jobs.resolve({jobs});rounds[index].trash.resolve({jobs:trash});}};
}
async function flush() {await Promise.resolve();await Promise.resolve();await Promise.resolve();}
function memoryStorage() {const values=new Map();return {getItem:key=>values.get(key)||null,setItem:(key,value)=>values.set(key,value)};}
function apiTimers(f) {
  const timers=[];
  f.context.AbortController=AbortController;
  f.context.setTimeout=(callback,milliseconds)=>{const timer={callback,milliseconds,cleared:false};timers.push(timer);return timer;};
  f.context.clearTimeout=timer=>{timer.cleared=true;};
  return timers;
}
async function test(name,body) {await body();cases.push(name);}
async function main() {
  await test('input ID pins import to originating project even when selection changed',async()=>{
    const f=fixture();f.select(b);const upload=f.input(a);
    await f.context.importPackage(upload);
    assert.equal(f.calls[0].url,'/api/jobs/aaa111aaa111/colab-result');
    assert.equal(f.state.selectedJob,b.id);assert.equal(upload.disabled,false);assert.equal(upload.value,'');
  });
  await test('existing chooser can recover ownership from its closest project element',async()=>{
    const f=fixture();f.select(b);await f.context.importPackage(f.input(a,'result',false));
    assert.equal(f.calls[0].url,'/api/jobs/aaa111aaa111/colab-result');
  });
  await test('unowned or removed project chooser is never routed to selected project',async()=>{
    const f=fixture();f.select(b);
    const unowned={files:[new Blob(['zip'])],dataset:{import:'result'}};
    await f.context.importPackage(unowned);
    const removed=f.input({...a,id:'ccc333ccc333'});await f.context.importPackage(removed);
    assert.equal(f.calls.length,0);
  });
  await test('selection change after upload starts cannot redirect POST or auto-open stale review',async()=>{
    const f=fixture();f.select(a);const pending=deferred();let reviewLoads=0;
    f.context.api=async (url,options)=>{f.calls.push({url,options});return pending.promise;};
    f.context.loadReview=async()=>{reviewLoads++;};
    const upload=f.input(a,'review');const promise=f.context.importPackage(upload);
    const other=f.select(b);other.innerHTML='Project B';pending.resolve({});await promise;
    assert.equal(f.calls[0].url,'/api/jobs/aaa111aaa111/review');
    assert.equal(other.innerHTML,'Project B');assert.equal(reviewLoads,0);assert.equal(upload.disabled,false);
  });
  await test('review response cannot replace another selected project editor or state',async()=>{
    const f=fixture();f.select(a);const pending=deferred();f.context.api=()=>pending.promise;
    const promise=f.context.loadReview(a);const other=f.select(b);other.innerHTML='Project B';
    pending.resolve(payload('Stale A'));await promise;
    assert.equal(other.innerHTML,'Project B');assert.equal(f.state.review,null);
  });
  await test('A to B to A selection rejects response bound to removed A editor',async()=>{
    const f=fixture();f.select(a);const pending=deferred();f.context.api=()=>pending.promise;
    const promise=f.context.loadReview(a);f.select(b);const fresh=f.select(a);fresh.innerHTML='New A';
    pending.resolve(payload('Old A'));await promise;
    assert.equal(fresh.innerHTML,'New A');assert.equal(f.state.review,null);
  });
  await test('poll replacing same-project editor rejects stale response',async()=>{
    const f=fixture();f.select(a);const pending=deferred();f.context.api=()=>pending.promise;
    const promise=f.context.loadReview(a);const replacement=f.select(a);replacement.innerHTML='Replacement';
    pending.resolve(payload('Old editor'));await promise;
    assert.equal(replacement.innerHTML,'Replacement');assert.equal(f.state.review,null);
  });
  await test('latest review request wins when responses complete out of order',async()=>{
    const f=fixture(),editor=f.select(a),first=deferred(),second=deferred();let count=0;
    f.context.api=()=>++count===1?first.promise:second.promise;
    const old=f.context.loadReview(a),fresh=f.context.loadReview(a);
    second.resolve(payload('New review'));await fresh;first.resolve(payload('Old review'));await old;
    assert.ok(editor.innerHTML.includes('New review'));assert.ok(!editor.innerHTML.includes('Old review'));
    assert.equal(f.state.review.translations['es-ES'][0].translated_text,'New review');
  });
  await test('stale review rejection never toasts against a different project',async()=>{
    const f=fixture();f.select(a);const pending=deferred();f.context.api=()=>pending.promise;
    const promise=f.context.loadReview(a);f.select(b);pending.reject(new Error('Late A error'));await promise;
    assert.deepEqual(f.toasts,[]);
  });
  await test('review save reads only owned editor inputs and guards changed selection',async()=>{
    const f=fixture(),editor=f.select(a);
    f.state.review={jobId:a.id,payload:payload('Before'),translations:payload('Before').translations};
    editor.inputs=[{dataset:{reviewLanguage:'es-ES',reviewIndex:'0'},value:'Owned edit'}];
    f.document.querySelectorAll=()=>[{dataset:{reviewLanguage:'es-ES',reviewIndex:'0'},value:'Alien edit'}];
    const button={disabled:false};await f.context.saveReview(a,button);
    assert.equal(f.calls[0].url,'/api/jobs/aaa111aaa111/review-approved');
    assert.equal(JSON.parse(f.calls[0].options.body).translations['es-ES'][0].translated_text,'Owned edit');
    assert.equal(button.disabled,false);f.calls.length=0;f.select(b);await f.context.saveReview(a,button);assert.equal(f.calls.length,0);
  });
  await test('rendered import controls receive fixed project ID and untouched polling keeps player',async()=>{
    const f=fixture();f.select(a);
    Object.defineProperty(f.container,'innerHTML',{configurable:true,set(markup){
      const id=/data-current-job="([^"]+)"/.exec(markup)[1];
      const controls=[{dataset:{import:'result'}},{dataset:{import:'review'}}];
      f.detail={dataset:{currentJob:id},imports:controls,querySelector:()=>null,querySelectorAll:selector=>selector==='[data-import]'?controls:[]};
    }});
    f.context.renderJobDetail(a);assert.equal(f.detail.imports.length,2);
    for(const input of f.detail.imports) assert.equal(input.dataset.importJob,a.id);
    let writes=0;const message={textContent:''};const current=f.detail;
    current.querySelector=selector=>selector==='[data-job-message]'?message:null;
    Object.defineProperty(f.container,'innerHTML',{configurable:true,set(){writes++;}});
    f.context.renderJobDetail({...a,message:'New message'});
    assert.equal(writes,0);assert.equal(message.textContent,'New message');
  });
  await test('overlapping refreshes coalesce and all callers wait for newest snapshot',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f);f.state.selectedJob=b.id;
    let firstDone=false,secondDone=false;
    const first=f.context.loadJobs({quiet:true}).then(()=>{firstDone=true;});
    const second=f.context.loadJobs({quiet:true,invalidate:true}).then(()=>{secondDone=true;});
    f.context.loadJobs({quiet:true});
    queue.complete(0,[a]);await flush();
    assert.equal(queue.rounds.length,2);assert.deepEqual(f.rendered,[]);
    assert.equal(f.state.selectedJob,b.id);assert.equal(firstDone,false);assert.equal(secondDone,false);
    queue.complete(1,[b,a]);await Promise.all([first,second]);
    assert.deepEqual(f.rendered,[b.id]);assert.equal(f.state.selectedJob,b.id);
    assert.equal(f.state.loadingJobs,false);assert.equal(queue.rounds.length,2);
  });
  await test('passive 4-second ticks never invalidate or indefinitely extend a slow current GET',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f);f.state.selectedJob=a.id;
    for(let round=0;round<8;round++) {
      let done=false;
      const current=f.context.loadJobs({quiet:true}).then(()=>{done=true;});
      // Two simulated 4-second ticks while this GET is still pending.
      f.context.loadJobs({quiet:true});f.context.loadJobs({quiet:true});
      queue.complete(round,[{...a,status:'review_required',message:`Snapshot ${round}`}]);
      await flush();
      assert.equal(f.state.jobs[0].message,`Snapshot ${round}`,'quiet polling must not discard a usable current snapshot');
      assert.equal(queue.rounds.length,round+1,'quiet ticks must not create an endless follow-up chain');
      await current;assert.equal(done,true);assert.equal(f.state.loadingJobs,false);
    }
    assert.equal(f.rendered.length,8);
  });
  await test('review import waits for one post-mutation snapshot despite passive ticks and then opens editor',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f),libraryApi=f.context.api,posted=deferred();f.select(a);
    let reviewLoads=0;f.context.loadReview=async job=>{assert.equal(job.id,a.id);reviewLoads++;};
    f.context.api=(url,options)=>options?.method==='POST'?posted.promise:libraryApi(url,options);
    const old=f.context.loadJobs({quiet:true}),upload=f.input(a,'review'),imported=f.context.importPackage(upload);
    f.context.loadJobs({quiet:true});posted.resolve({});await flush();
    queue.complete(0,[a]);await flush();
    assert.equal(queue.rounds.length,2);assert.deepEqual(f.rendered,[]);
    f.context.loadJobs({quiet:true});f.context.loadJobs({quiet:true});
    queue.complete(1,[{...a,status:'review_required'}]);await Promise.all([old,imported]);
    assert.equal(f.state.jobs[0].status,'review_required');assert.equal(reviewLoads,1);
    assert.equal(upload.disabled,false);assert.equal(queue.rounds.length,2);
  });
  await test('approved review download is requested once immediately after POST and quiet ticks cannot hold the button',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f),libraryApi=f.context.api,posted=deferred();f.select(a);
    f.state.review={jobId:a.id,payload:payload('Approved'),translations:payload('Approved').translations};
    f.context.api=(url,options)=>options?.method==='POST'?posted.promise:libraryApi(url,options);
    const old=f.context.loadJobs({quiet:true}),button={disabled:false},saved=f.context.saveReview(a,button);
    f.context.loadJobs({quiet:true});posted.resolve({download_url:'/approved-synthetic.zip'});await flush();
    assert.deepEqual(f.downloads,['/approved-synthetic.zip']);
    queue.complete(0,[a]);await flush();assert.equal(queue.rounds.length,2);
    f.context.loadJobs({quiet:true});f.context.loadJobs({quiet:true});
    queue.complete(1,[{...a,status:'review_approved'}]);await Promise.all([old,saved]);
    assert.equal(f.state.jobs[0].status,'review_approved');assert.equal(button.disabled,false);
    assert.deepEqual(f.downloads,['/approved-synthetic.zip']);assert.equal(queue.rounds.length,2);
  });
  await test('create while old library is loading never displays an old project instead',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f),created={...b,status:'preparing',has_colab_package:false};
    const libraryApi=f.context.api;
    f.context.api=(url,options)=>url==='/api/jobs'&&options?.method==='POST'?Promise.resolve({job:created}):libraryApi(url,options);
    const fields={videoInput:{files:[new Blob(['synthetic source'])]},audioInput:{files:[]},referenceInput:{files:[]},voiceConsent:{checked:true},createButton:{disabled:false},createButtonLabel:{textContent:''},sourceLanguage:{value:'pt'},qualityProfile:{value:'quality'},reviewMode:{value:'automatic'},requireLipsync:{checked:true},preserveAmbience:{checked:true},voiceText:{value:''},speakerCount:{value:'1'},subtitleMode:{value:'none'},sourcePreview:{videoWidth:1920,videoHeight:1080},faceSelection:{value:'auto'}};
    for(const [id,element] of Object.entries(fields)) f.elements[`#${id}`]=element;
    f.context.window={open:()=>null};
    f.context.showView=view=>{f.state.view=view;f.context.loadJobs({quiet:true});};
    const old=f.context.loadJobs({quiet:true}),creation=f.context.startProject({preventDefault(){}});
    await flush();assert.equal(f.state.selectedJob,created.id);
    assert.deepEqual(f.rendered,[created.id]);assert.equal(f.state.jobs[0].id,created.id);
    queue.complete(0,[a]);await flush();assert.equal(queue.rounds.length,2);
    assert.ok(!f.rendered.includes(a.id));assert.equal(f.state.selectedJob,created.id);
    queue.complete(1,[a]);await Promise.all([old,creation]);
    assert.equal(f.state.selectedJob,created.id);assert.equal(f.state.jobs[0].id,created.id);
    assert.ok(!f.rendered.includes(a.id));assert.equal(fields.createButton.disabled,false);
    const confirmed=f.context.loadJobs({quiet:true});queue.complete(2,[created,a]);await confirmed;
    assert.equal(f.state.jobs.filter(job=>job.id===created.id).length,1);
    assert.equal(f.state.launchPending.job,null);
  });
  await test('archiving a provisional project removes it and cancels deferred Colab launch',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f),libraryApi=f.context.api;f.state.selectedJob=b.id;
    f.state.launchPending={id:b.id,job:b,tab:null,url:''};f.context.window={confirm:()=>true};
    f.context.api=(url,options)=>options?.method==='DELETE'?Promise.resolve({}):libraryApi(url,options);
    const button={dataset:{action:'archive'},disabled:false};
    const archive=f.context.jobAction({target:{closest:()=>button}});await flush();
    assert.equal(f.state.launchPending,null);queue.complete(0,[a]);await archive;
    assert.equal(f.state.selectedJob,a.id);assert.equal(f.state.jobs.some(job=>job.id===b.id),false);
    assert.deepEqual(f.downloads,[]);
  });
  await test('archive acknowledgement invalidates an already pending pre-archive GET',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f),libraryApi=f.context.api;f.state.selectedJob=b.id;
    f.context.window={confirm:()=>true};
    f.context.api=(url,options)=>options?.method==='DELETE'?Promise.resolve({}):libraryApi(url,options);
    const old=f.context.loadJobs({quiet:true}),button={dataset:{action:'archive'},disabled:false};
    const archived=f.context.jobAction({target:{closest:()=>button}});await flush();
    f.context.loadJobs({quiet:true});queue.complete(0,[b,a]);await flush();
    assert.deepEqual(f.rendered,[]);assert.equal(queue.rounds.length,2);
    queue.complete(1,[a]);await Promise.all([old,archived]);
    assert.equal(f.state.selectedJob,a.id);assert.equal(f.state.jobs.some(job=>job.id===b.id),false);
  });
  await test('trash acknowledgement invalidates pending live snapshot without resurrecting removed project',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f),libraryApi=f.context.api;f.state.selectedJob=b.id;
    f.context.confirmProjectAction=async()=>true;
    f.context.api=(url,options)=>options?.method==='POST'?Promise.resolve({}):libraryApi(url,options);
    const old=f.context.loadJobs({quiet:true}),button={dataset:{action:'trash'},disabled:false};
    const trashed=f.context.jobAction({target:{closest:()=>button}});await flush();
    f.context.loadJobs({quiet:true});queue.complete(0,[b,a]);await flush();
    assert.deepEqual(f.rendered,[]);assert.equal(queue.rounds.length,2);
    queue.complete(1,[a],[{...b,status:'deleted'}]);await Promise.all([old,trashed]);
    assert.equal(f.state.selectedJob,a.id);assert.equal(f.state.jobs.some(job=>job.id===b.id),false);
    assert.equal(f.state.trashJobs[0].id,b.id);assert.equal(f.state.trashJobs[0].status,'deleted');
  });
  await test('quiet polling cannot hide errors from an overlapping explicit refresh',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f);
    const quiet=f.context.loadJobs({quiet:true}),explicit=f.context.loadJobs();
    queue.complete(0,[a]);await flush();
    queue.rounds[1].jobs.reject(new Error('Latest refresh failed'));queue.rounds[1].trash.resolve({jobs:[]});
    await Promise.all([quiet,explicit]);assert.deepEqual(f.toasts,['Latest refresh failed']);assert.equal(f.state.loadingJobs,false);
    const recovery=f.context.loadJobs({quiet:true});queue.complete(2,[b]);await recovery;
    assert.equal(f.state.selectedJob,b.id);assert.equal(f.state.loadingJobs,false);
  });
  await test('failed superseded request still drains queued refresh without stale error',async()=>{
    const f=libraryFixture(),queue=queuedLibrary(f);
    const first=f.context.loadJobs(),second=f.context.loadJobs({quiet:true,invalidate:true});
    queue.rounds[0].jobs.reject(new Error('Superseded error'));queue.rounds[0].trash.resolve({jobs:[]});await flush();
    assert.equal(queue.rounds.length,2);queue.complete(1,[b]);await Promise.all([first,second]);
    assert.deepEqual(f.toasts,[]);assert.deepEqual(f.rendered,[b.id]);
  });
  await test('same-tab reload restores selected project and validates it against current library',async()=>{
    const storage=memoryStorage(),before=libraryFixture({storage});before.state.selectedJob=b.id;before.context.renderJobs();
    const after=libraryFixture({storage});after.context.restoreLibrarySelection();after.context.renderJobs();
    assert.equal(after.state.selectedJob,b.id);assert.deepEqual(after.rendered,[b.id]);
    const removed=libraryFixture({storage});removed.state.jobs=[a];removed.context.restoreLibrarySelection();removed.context.renderJobs();
    assert.equal(removed.state.selectedJob,a.id);assert.deepEqual(removed.rendered,[a.id]);
  });
  await test('same-tab reload keeps trash selection separate and cannot overwrite live selection',async()=>{
    const storage=memoryStorage(),deleted={...b,status:'deleted'},before=libraryFixture({storage});
    before.state.librarySection='trash';before.state.trashJobs=[deleted];before.state.selectedJob=b.id;before.context.renderJobs();
    const after=libraryFixture({storage});after.state.trashJobs=[deleted];after.context.restoreLibrarySelection();after.context.renderJobs();
    assert.equal(after.state.librarySection,'trash');assert.equal(after.state.selectedJob,b.id);
    const live=libraryFixture({storage});live.state.selectedJob=a.id;live.context.restoreLibrarySelection();
    assert.equal(live.state.selectedJob,a.id);assert.equal(live.state.librarySection,'projects');
  });
  await test('unavailable or corrupt session storage never breaks library or refresh',async()=>{
    const corrupt=memoryStorage();corrupt.setItem('dubstudio.librarySelection','{invalid');
    const f=libraryFixture({storage:corrupt});f.context.restoreLibrarySelection();f.context.renderJobs();assert.equal(f.state.selectedJob,a.id);
    const inaccessible=libraryFixture({storage:{getItem(){throw new Error('Denied');},setItem(){throw new Error('Denied');}}});
    inaccessible.context.restoreLibrarySelection();inaccessible.context.renderJobs();assert.equal(inaccessible.state.selectedJob,a.id);
    const invalid=memoryStorage();invalid.setItem('dubstudio.librarySelection',JSON.stringify({section:'outside',jobId:b.id}));
    const ignored=libraryFixture({storage:invalid});ignored.context.restoreLibrarySelection();assert.equal(ignored.state.selectedJob,null);
  });
  await test('production API GET deadline is 45 seconds and successful requests clear it',async()=>{
    const f=fixture(),timers=apiTimers(f);let requested;
    f.context.fetch=async (url,options)=>{requested={url,options};return {ok:true,json:async()=>({ready:true})};};
    const answer=await f.productionApi('/api/jobs');
    assert.equal(answer.ready,true);assert.equal(timers[0].milliseconds,45000);assert.equal(timers[0].cleared,true);
    assert.equal(requested.options.cache,'no-store');assert.equal(requested.options.signal.aborted,false);
    assert.equal('timeoutMs' in requested.options,false);
  });
  await test('production API bounds a never-settling GET and aborts without retry',async()=>{
    const f=fixture(),timers=apiTimers(f);let count=0,signal;
    f.context.fetch=(_url,options)=>{count++;signal=options.signal;return new Promise(()=>{});};
    const pending=f.productionApi('/api/jobs'),failed=assert.rejects(pending,/45 segundos/);
    timers[0].callback();await failed;
    assert.equal(signal.aborted,true);assert.equal(count,1);assert.equal(timers[0].cleared,true);
  });
  await test('production API deadline also bounds a stalled JSON body after 200 headers',async()=>{
    const f=fixture(),timers=apiTimers(f);
    f.context.fetch=async()=>({ok:true,json:()=>new Promise(()=>{})});
    const pending=f.productionApi('/api/jobs');await flush();
    const failed=assert.rejects(pending,/45 segundos/);timers[0].callback();await failed;
  });
  await test('production API POST deadline explains uncertain server state and never resends',async()=>{
    const f=fixture(),timers=apiTimers(f);let count=0;
    f.context.fetch=()=>{count++;return new Promise(()=>{});};
    const pending=f.productionApi('/api/jobs/aaa111aaa111/review-approved',{method:'POST',body:'{}'});
    assert.equal(timers[0].milliseconds,120000);
    const failed=assert.rejects(pending,/pode ter sido salva.*antes de repetir.*Nenhum reenvio automático/);
    timers[0].callback();await failed;assert.equal(count,1);
  });
  await test('large multipart import deadline scales with known file bytes without allocating personal media',async()=>{
    const f=fixture(),timers=apiTimers(f),form=Object.create(FormData.prototype);
    // Size-only fake FormData values: no 150MiB file or browser/network involved.
    form.values=function*(){yield {size:150*1024*1024};yield 'synthetic option';};
    f.context.fetch=async()=>({ok:true,json:async()=>({received:true})});
    await f.productionApi('/api/jobs/aaa111aaa111/colab-result',{method:'POST',body:form});
    assert.equal(timers[0].milliseconds,270000);assert.equal(timers[0].cleared,true);
  });
  await test('custom API deadline stays bounded and is not leaked as a fetch option',async()=>{
    const f=fixture(),timers=apiTimers(f);let calls=0,options;
    f.context.fetch=async(_url,request)=>{calls++;options=request;return {ok:true,json:async()=>({})};};
    await f.productionApi('/api/jobs',{timeoutMs:60000});assert.equal(timers[0].milliseconds,60000);
    assert.equal('timeoutMs' in options,false);
    for(const invalid of ( [true,0,Infinity,1800001] )) await assert.rejects(f.productionApi('/api/jobs',{timeoutMs:invalid}),/Prazo/);
    assert.equal(calls,1);
  });
  console.log(`Async project ownership and review response regressions: PASS (${cases.length} scenarios)`);
}
main().catch(error=>{console.error(error.stack||error);process.exitCode=1;});
