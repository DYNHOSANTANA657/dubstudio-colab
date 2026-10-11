'use strict';
// Exercise the production action handler offline; never touch user projects.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const notifications = [], requests = [], dialogs = [];
class Element {
  constructor(tagName) { this.tagName = tagName.toUpperCase(); this.children = []; this.listeners = {}; this.attributes = {}; this.removed = false; }
  setAttribute(name, value) { this.attributes[name] = value; }
  append(...children) { this.children.push(...children); children.forEach(child => { child.parent = this; }); }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  emit(name, event = {}) { (this.listeners[name] || []).forEach(listener => listener(event)); }
  focus() { document.activeElement = this; }
  showModal() { this.open = true; }
  close() { this.open = false; this.emit('close'); }
  remove() { this.removed = true; }
  get text() { return [this.textContent || '', ...this.children.map(child => child.text)].join('\n'); }
}
const previousFocus = new Element('button');
const document = {activeElement: previousFocus, body: new Element('body'),
  createElement(tagName) { const element = new Element(tagName); if (tagName === 'dialog') dialogs.push(element); return element; },
  querySelector(selector) { return selector === '#projectActionDialog' ? dialogs.find(dialog => !dialog.removed) || null : null; }};
const context = vm.createContext({console, document,
  window: {confirm() { throw new Error('Trash/restore must never open a native JS dialog'); }}});
vm.runInContext(source.replace(/init\(\);\s*$/, ''), context);
const state = vm.runInContext('state', context);
context.api = async (url, options) => { requests.push({url, options}); return {ok: true}; };
context.loadJobs = async () => {};
context.toast = value => notifications.push(value);
const job = {id: '0123456789ab', status: 'processing_colab', options: {operation: 'dub_video'}, metadata: {original_name: 'video.mp4'}};
const button = {dataset: {action: 'trash'}, disabled: false};
const event = {target: {closest: () => button}};
(async () => {
  state.jobs = [job]; state.selectedJob = job.id;
  state.launchPending = {id: job.id};
  let pending = context.jobAction(event);
  let dialog = dialogs.at(-1);
  assert.equal(dialog.open, true);
  assert.equal(dialog.attributes['aria-labelledby'], 'projectActionTitle');
  assert.equal(dialog.children[0].tagName, 'H2');
  assert.ok(dialog.text.includes('Downloads e arquivos originais fora do Studio ficam intactos'));
  assert.ok(dialog.text.includes('NÃO cancela a execução no Colab'));
  assert.equal(document.activeElement.textContent, 'Cancelar', 'Focus must default to cancellation');
  dialog.children.at(-1).children[0].emit('click');
  await pending;
  assert.equal(requests.length, 0, 'Cancel must not send any mutation');
  assert.equal(button.disabled, false);
  assert.equal(dialog.removed, true);
  assert.equal(document.activeElement, previousFocus);
  pending = context.jobAction(event);
  dialog = dialogs.at(-1);
  dialog.children.at(-1).children[1].emit('click');
  dialog.emit('close'); // A second close must not resolve/cancel the accepted action again.
  await pending;
  assert.equal(requests[0].url, `/api/jobs/${job.id}/trash`);
  assert.equal(requests[0].options.method, 'POST');
  assert.deepEqual(JSON.parse(requests[0].options.body), {confirm: true});
  assert.equal(state.selectedJob, null);
  assert.equal(state.launchPending, null, 'Exclude must not launch a notebook later');
  assert.ok(notifications[0].includes('Lixeira'));

  const trashed = {...job, status: 'deleted'};
  state.librarySection = 'trash'; state.trashJobs = [trashed]; state.selectedJob = job.id;
  assert.deepEqual(context.visibleJobs(), state.trashJobs);
  button.dataset.action = 'restore'; button.disabled = false;
  pending = context.jobAction(event);
  dialog = dialogs.at(-1);
  assert.equal(dialog.children.at(-1).children[1].textContent, 'Restaurar projeto');
  assert.ok(dialog.text.includes('não inicia a execução no Colab'));
  dialog.children.at(-1).children[1].emit('click');
  await pending;
  assert.equal(requests[1].url, `/api/trash/${job.id}/restore`);
  assert.deepEqual(JSON.parse(requests[1].options.body), {confirm: true});
  assert.equal(state.librarySection, 'projects');
  assert.equal(state.selectedJob, job.id);

  // Escape is an explicit cancellation, including late clicks on a detached node.
  pending = context.jobAction({...event, target: {closest: () => ({dataset: {action: 'trash'}, disabled: false})}});
  dialog = dialogs.at(-1);
  let prevented = false;
  dialog.emit('cancel', {preventDefault() { prevented = true; }});
  dialog.children.at(-1).children[1].emit('click');
  await pending;
  assert.equal(prevented, true);
  assert.equal(requests.length, 2, 'Esc and late approval cannot mutate anything');
  assert.equal(dialog.removed, true);

  // User titles are plain text, never HTML, even inside the risk confirmation.
  const hostile = {...job, metadata: {original_name: '<img src=x onerror=alert(1)>'}};
  pending = context.confirmProjectAction(hostile);
  dialog = dialogs.at(-1);
  assert.ok(dialog.children[1].textContent.includes('<img src=x onerror=alert(1)>'));
  assert.equal(dialog.children[1].children.length, 0);
  assert.equal(await context.confirmProjectAction(hostile), false, 'Do not stack concurrent modals');
  dialog.children.at(-1).children[0].emit('click');
  assert.equal(await pending, false);

  // A failed mutation reports the error and leaves the current project usable.
  context.api = async () => { throw new Error('test permission failure'); };
  button.dataset.action = 'trash'; button.disabled = false;
  pending = context.jobAction(event);
  dialog = dialogs.at(-1);
  dialog.children.at(-1).children[1].emit('click');
  await pending;
  assert.equal(button.disabled, false);
  assert.equal(state.selectedJob, job.id);
  assert.equal(notifications.at(-1), 'test permission failure');
  console.log('Project trash: accessible modal, safe focus, cancel/Esc, one resolution, plain-text titles, external-run warning, restore and failure preservation: PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
