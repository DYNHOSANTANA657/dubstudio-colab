'use strict';
// Offline UI regression: execute production functions without starting the app.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
assert.ok(source.endsWith('init();\n') || source.endsWith('init();\r\n'));
let detail;
const context = vm.createContext({document: {querySelector: selector => selector === '#jobDetail' ? {
  querySelector: () => detail
} : null}, console});
vm.runInContext(source.replace(/init\(\);\s*$/, ''), context);
const state = vm.runInContext('state', context);
const output = (language, status='complete', files={video:{},wav:{}}) => ({language,status,files});
const job = {id:'0123456789ab',options:{operation:'dub_video'},outputs:[output('pl-PL'),output('en-US'),output('es-ES'),output('de-DE','failed')]};
const tts = {id:'abcdef012345',options:{operation:'text_to_voice'},outputs:[output('es-ES','complete',{wav:{},mp3:{}})]};
assert.equal(context.previewOutputs(job).length,3);
assert.equal(context.selectedPreview(job).language,'pl-PL');
state.previewLanguages.set(job.id,'es-ES');
assert.equal(context.selectedPreview(job).language,'es-ES');
assert.equal(context.selectedPreview(tts).kind,'wav');
assert.equal(context.selectedPreview(job).language,'es-ES','selection is per project');
state.previewLanguages.set(job.id,'de-DE');
assert.equal(context.selectedPreview(job).language,'pl-PL','failed versions are not previewable');
const attrs = {src:'/api/jobs/0123456789ab/outputs/pl-PL/video?preview=1'};
const player = {tagName:'VIDEO', volume:1, muted:false, paused:0, loaded:0,
  getAttribute:name=>attrs[name], setAttribute:(name,value)=>{attrs[name]=value;},
  pause(){this.paused++;}, load(){this.loaded++;}};
const selector = {value:'pl-PL'};
detail = {dataset:{currentJob:job.id},querySelector:name=>name==='[data-result-preview]'?player:selector};
assert.equal(context.selectJobPreview(job,'es-ES'),true);
assert.equal(attrs.src,'/api/jobs/0123456789ab/outputs/es-ES/video?preview=1');
assert.equal(selector.value,'es-ES');
assert.equal(player.paused,1);
assert.equal(player.loaded,1);
assert.equal(context.selectJobPreview(job,'es-ES'),true);
assert.equal(player.loaded,1,'poll/same-language selection must not reset playback');
assert.equal(context.selectJobPreview(job,'de-DE'),false);
detail.dataset.currentJob='different';
assert.equal(context.selectJobPreview(job,'en-US'),false);
assert.equal(attrs.src,'/api/jobs/0123456789ab/outputs/es-ES/video?preview=1');
assert.ok(context.renderJobPreview(tts).includes('<audio controls'));
assert.ok(context.renderJobPreview(job).includes('Idioma da prévia'));
const facialOutput = {...output('es-ES'),validation:{lipsync_confirmed:true}};
const facialMarkup = context.renderOutput(job,facialOutput);
assert.ok(facialMarkup.includes('Lip-sync aplicado'));
assert.ok(!facialMarkup.includes('Lip-sync confirmado'),'machine evidence must not imply perceptual certification');
console.log('Preview selection, per-project state, playback preservation and safe output filtering: PASS');
