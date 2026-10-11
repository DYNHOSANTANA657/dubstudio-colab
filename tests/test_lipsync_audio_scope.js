'use strict';
// Execute production view labels in a minimal DOM, with no browser or request.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const nodes = new Map();
function node(selector) {
  if (!nodes.has(selector)) nodes.set(selector, {
    value: selector === '#faceSelection' ? 'auto' : selector === '#speakerCount' ? '3' : '',
    hidden: false, checked: false, disabled: false, textContent: '', innerHTML: '',
    options: [{textContent: ''}], classList: {add(){}, remove(){}, toggle(){}}, setAttribute(){}
  });
  return nodes.get(selector);
}
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const context = vm.createContext({console, document: {querySelector: node, querySelectorAll: () => []}});
vm.runInContext(source.replace(/init\(\);\s*$/, ''), context);
const state = vm.runInContext('state', context);
state.languages = new Set(['pl-PL', 'es-ES']);
state.faceSelection = {frame_time: 1, bbox: [.1, .1, .3, .4], normalized: true};
const preserved = {languages: [...state.languages], face: JSON.stringify(state.faceSelection)};
context.showView('lipsync', {updateHash: false});
assert.equal(state.operation, 'lipsync_video');
assert.ok(node('#pageDescription').textContent.includes('rosto detectado ou escolhido'));
assert.ok(!node('#pageDescription').textContent.includes('rosto falante'));
assert.ok(node('#lipsyncDescription').textContent.includes('um único mix'));
assert.ok(node('#lipsyncDescription').textContent.includes('Não identifica nem associa automaticamente'));
assert.equal(node('#speakerField').hidden, true, 'Speaker count applies only to dubbing');
assert.equal(node('#languageField').hidden, true, 'Uploaded audio is not translated');
assert.equal(node('#audioInput').required, true);
assert.equal(node('#requireLipsync').checked, true);
assert.equal(node('#requireLipsync').disabled, true);
assert.equal(context.usesLipsync(), true);
context.showView('dub', {updateHash: false});
assert.equal(state.operation, 'dub_video');
assert.equal(node('#speakerField').hidden, false);
assert.equal(node('#speakerCount').value, '3');
assert.equal(node('#lipsyncDescription').textContent, 'Ajustar boca à voz dublada');
state.dubLipsync = false;
context.showView('dub', {updateHash: false});
assert.equal(context.usesLipsync(), false);
assert.equal(node('#lipsyncDescription').textContent, 'Dublar áudio sem alterar movimentos da boca');
assert.ok(!node('#lipsyncDescription').textContent.includes('mix'));
context.showView('tts', {updateHash: false});
assert.equal(state.operation, 'text_to_voice');
assert.equal(node('#lipsyncRow').hidden, true);
assert.equal(node('#speakerField').hidden, true);
assert.equal(node('#referenceInput').required, true);
assert.equal(node('#voiceText').required, true);
assert.equal(node('#languageFieldLabel').textContent, 'Idiomas da pronúncia');
assert.ok(node('#languageNote').textContent.includes('sem tradução automática'));
assert.deepEqual([...state.languages], preserved.languages);
assert.equal(JSON.stringify(state.faceSelection), preserved.face);
context.showView('lipsync', {updateHash: false});
assert.ok(node('#lipsyncDescription').textContent.includes('várias vozes a vários rostos'));
assert.equal(node('#requireLipsync').checked, true, 'Returning from optional dubbing cannot disable uploaded lip-sync');
assert.equal(state.dubLipsync, false, 'Operation wording must not reset dubbing preferences');
console.log('PASS: uploaded-audio mix limits, unchanged dubbing/TTS controls, retained selections and lip-sync semantics');
