'use strict';

const state = {
  view: 'dub', operation: 'dub_video', catalog: { languages: [], presets: [] },
  languages: new Set(['pl-PL']), jobs: [], selectedJob: null, settings: {},
  health: null, loadingJobs: false, jobsLoadPromise: null, jobsRefreshRequested: false, jobsRefreshQuiet: true,
  launchPending: null, previewUrl: null, review: null, reviewRequest: 0,
  faceSelection: null, faceDrag: null, dubLipsync: true, previewLanguages: new Map(),
  trashJobs: [], librarySection: 'projects'
};
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const icons = {
  translate: '<path d="M3 5h12M9 3v2M5 5c0 5 5 9 8 10M13 5c0 5-5 9-9 10M14 21l4-10 4 10M15.5 17h5"/>',
  face: '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3"/><path d="M8 9h.01M16 9h.01M8 15c2 2 6 2 8 0M12 10v3"/>',
  wave: '<path d="M3 10v4M7 6v12M12 3v18M17 7v10M21 10v4"/>',
  folder: '<path d="M3 6a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6Z"/>',
  settings: '<path d="m10 3-.5 2.2-2 .9L5.6 5 3 9l1.6 1.5v3L3 15l2.6 4 1.9-1.1 2 .9L10 21h4l.5-2.2 2-.9 1.9 1.1 2.6-4-1.6-1.5v-3L21 9l-2.6-4-1.9 1.1-2-.9L14 3Z"/><circle cx="12" cy="12" r="3"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/>',
  video: '<rect x="3" y="5" width="14" height="14" rx="3"/><path d="m17 10 4-3v10l-4-3"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
  download: '<path d="M12 3v13m-5-5 5 5 5-5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
  spark: '<path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6Z"/><path d="M20 3v4M18 5h4"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/>',
  globe: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
  audio: '<path d="M9 18V5l11-2v13M9 9l11-2"/><ellipse cx="6" cy="18" rx="3" ry="2"/><ellipse cx="17" cy="16" rx="3" ry="2"/>',
  mic: '<rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  external: '<path d="M13 3h8v8m0-8L10 14M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"/>',
  refresh: '<path d="M20 7v5h-5M4 17v-5h5M6 6a8 8 0 0 1 13 2M5 16a8 8 0 0 0 13 2"/>',
  search: '<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  colab: '<path d="M10 7a6 6 0 1 0 0 10M14 7a6 6 0 1 1 0 10M8 12h8"/>'
};
function paintIcons(root = document) {
  $$('[data-icon]', root).forEach(node => {
    node.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[node.dataset.icon] || icons.spark}</svg>`;
  });
}
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
async function api(path, options = {}) {
  const { timeoutMs: requestedTimeout, signal: callerSignal, ...requestOptions } = options;
  const readOnly = ['GET', 'HEAD'].includes(String(options.method || 'GET').toUpperCase());
  let timeoutMs = readOnly ? 45000 : 120000;
  if (!readOnly && options.body instanceof FormData) {
    let bytes = 0;
    for (const value of options.body.values()) {
      if (value && typeof value.size === 'number' && Number.isFinite(value.size) && value.size > 0) bytes += value.size;
    }
    // Allow local ZIP upload plus extraction/validation, conservatively at
    // 1 MiB/s, while still bounding a stalled import to thirty minutes.
    timeoutMs = Math.min(1800000, 120000 + Math.ceil(bytes / (1024 * 1024)) * 1000);
  }
  if (requestedTimeout !== undefined) {
    if (typeof requestedTimeout !== 'number' || !Number.isFinite(requestedTimeout) || requestedTimeout < 1000 || requestedTimeout > 1800000) throw new Error('Prazo da solicitação inválido.');
    timeoutMs = requestedTimeout;
  }
  const controller = new AbortController();
  const forwardAbort = () => controller.abort(callerSignal.reason);
  if (callerSignal?.aborted) forwardAbort();
  else callerSignal?.addEventListener('abort', forwardAbort, { once: true });
  let expired = false, timer;
  const timeoutError = new Error(readOnly
    ? `O Studio não respondeu em ${Math.round(timeoutMs / 1000)} segundos. Atualize a lista e tente novamente.`
    : 'O Studio não concluiu a resposta dentro do prazo. A operação pode ter sido salva; confira o projeto antes de repetir o envio. Nenhum reenvio automático foi feito.');
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => { expired = true; controller.abort(); reject(timeoutError); }, timeoutMs);
  });
  try {
    const operation = (async () => {
      const response = await fetch(path, { cache: 'no-store', ...requestOptions, signal: controller.signal });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || payload.message || `A solicitação falhou (${response.status}).`);
      return payload;
    })();
    return await Promise.race([operation, deadline]);
  } catch (error) { throw expired ? timeoutError : error; }
  finally { clearTimeout(timer); callerSignal?.removeEventListener('abort', forwardAbort); }
}
function toast(message) {
  $('#toast').textContent = message;
  $('#toast').classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $('#toast').classList.remove('show'), 6500);
}
function formatDuration(value) {
  const seconds = Math.max(0, Math.round(Number(value || 0)));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}
function formatBytes(value) {
  let size = Number(value || 0), index = 0;
  while (size >= 1024 && index < 3) { size /= 1024; index += 1; }
  return `${size.toFixed(index ? 1 : 0)} ${['B', 'KB', 'MB', 'GB'][index]}`;
}
function languageName(code) { return code === 'und' ? 'Áudio enviado' : state.catalog.languages.find(language => language.code === code)?.name || code; }
function notebookUrl() {
  const value = state.settings.colab?.notebook_url || '';
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && url.hostname === 'colab.research.google.com' ? url.href : '';
  } catch { return ''; }
}
function jobPath(id, suffix = '') { return `/api/jobs/${encodeURIComponent(id)}${suffix}`; }
function outputPath(job, language, kind, preview = false) {
  return jobPath(job.id, `/outputs/${encodeURIComponent(language)}/${encodeURIComponent(kind)}${preview ? '?preview=1' : ''}`);
}
const labels = {
  dub: { operation: 'dub_video', crumb: 'Dublar vídeo', kicker: 'UM VÍDEO. MUITAS POSSIBILIDADES.', title: 'Sua voz, em novos idiomas', description: 'Duble seu vídeo com o timbre original e sincronização labial.', media: 'Seu vídeo original', controls: 'Personalize a dublagem' },
  lipsync: { operation: 'lipsync_video', crumb: 'Lip-sync com áudio', kicker: 'VOZ E IMAGEM. EM SINTONIA.', title: 'Dê o ritmo certo à sua criação', description: 'Aplique o áudio enviado a um rosto detectado ou escolhido por você.', media: 'Seu vídeo', controls: 'Personalize o lip-sync' },
  tts: { operation: 'text_to_voice', crumb: 'Texto para voz', kicker: 'PALAVRAS COM A SUA IDENTIDADE.', title: 'Transforme seu texto em voz', description: 'Crie uma narração com uma referência vocal autorizada.', media: 'Seu texto', controls: 'Personalize a voz' }
};
function showView(view, { updateHash = true } = {}) {
  if (!['dub', 'lipsync', 'tts', 'creations', 'settings'].includes(view)) view = 'dub';
  state.view = view;
  $$('.nav-item').forEach(node => {
    node.classList.toggle('active', node.dataset.view === view);
    node.setAttribute('aria-label', node.textContent.trim());
    node.setAttribute('aria-current', node.dataset.view === view ? 'page' : 'false');
  });
  $$('.view').forEach(node => node.classList.remove('active'));
  if (labels[view]) {
    const config = labels[view];
    state.operation = config.operation;
    $('#createView').classList.add('active');
    $('#breadcrumbTitle').textContent = config.crumb;
    $('#pageTitle').innerHTML = `${escapeHtml(config.title)}<span>.</span>`;
    $('#operationKicker').textContent = config.kicker;
    $('#pageDescription').textContent = config.description;
    $('#mediaHeading').textContent = config.media;
    $('#controlsHeading').textContent = config.controls;
    $('#videoArea').hidden = view === 'tts';
    $('#textArea').hidden = view !== 'tts';
    $('#mediaFormat').hidden = view === 'tts';
    $('#dubFields').hidden = view !== 'dub';
    $('#languageField').hidden = view === 'lipsync';
    $('#audioField').hidden = view !== 'lipsync';
    $('#referenceField').hidden = view !== 'tts';
    $('#voiceIdentity').hidden = view !== 'dub';
    $('#lipsyncRow').hidden = view === 'tts';
    $('#requireLipsync').disabled = view !== 'dub';
    $('#requireLipsync').checked = view === 'lipsync' || (view === 'dub' && state.dubLipsync);
    $('#ambienceRow').hidden = view !== 'dub';
    $('#speakerField').hidden = view !== 'dub';
    $('#reviewField').hidden = view !== 'dub';
    $('#subtitleField').hidden = view !== 'dub';
    $('#faceField').hidden = !usesLipsync();
    $('#mediaBenefits').hidden = view !== 'dub';
    $('#faceOverlay').hidden = !usesLipsync() || $('#faceSelection').value !== 'manual';
    $('#videoInput').required = view !== 'tts';
    $('#audioInput').required = view === 'lipsync';
    $('#referenceInput').required = view === 'tts';
    $('#voiceText').required = view === 'tts';
    $('#languageFieldLabel').textContent = view === 'tts' ? 'Idiomas da pronúncia' : 'Idiomas de destino';
    $('#languageNote').textContent = view === 'tts' ? 'Seu texto será narrado como foi escrito, sem tradução automática. Escolha o idioma correspondente ao texto.' : 'Somente idiomas compatíveis com timbre original. Idioma e sotaque regional são capacidades diferentes.';
    updateLipsyncOptions();
  } else {
    $(`#${view}View`).classList.add('active');
    $('#breadcrumbTitle').textContent = view === 'creations' ? 'Minhas criações' : 'Configurações';
    if (view === 'creations') loadJobs({ quiet: true });
    if (view === 'settings') loadSettings();
  }
  if (updateHash) history.replaceState(null, '', `#${view}`);
}
function usesLipsync() {
  return state.operation === 'lipsync_video' || (state.operation === 'dub_video' && $('#requireLipsync').checked);
}
function updateLipsyncOptions() {
  const enabled = usesLipsync();
  $('#lipsyncLabel').textContent = state.operation === 'lipsync_video' ? 'Sincronização labial' : 'Sincronização labial (opcional)';
  $('#lipsyncDescription').textContent = state.operation === 'lipsync_video' ? 'O áudio é usado como um único mix. Não identifica nem associa automaticamente várias vozes a vários rostos.' : enabled ? 'Ajustar boca à voz dublada' : 'Dublar áudio sem alterar movimentos da boca';
  $('#lipBenefit').textContent = enabled ? 'Voz e imagem no mesmo ritmo' : 'Movimentos originais preservados';
  $('#faceField').hidden = !enabled;
  $('#faceOverlay').hidden = !enabled || $('#faceSelection').value !== 'manual';
  $('#qualityProfile').options[0].textContent = state.operation === 'text_to_voice' ? 'Equilibrada · recomendada' : enabled ? 'Equilibrada · com lip-sync' : 'Equilibrada · somente dublagem';
  if (state.operation === 'dub_video') $('#pageDescription').textContent = enabled ? 'Duble seu vídeo com o timbre original e sincronização labial.' : 'Duble seu vídeo com o timbre original, sem alterar os movimentos da boca.';
}
function renderSelectedLanguages() {
  const languages = [...state.languages];
  $('#languageSummary').textContent = languages.length === 1 ? languageName(languages[0]) : languages.length ? `${languages.length} idiomas selecionados` : 'Escolha os idiomas';
  $('#selectedLanguages').innerHTML = languages.length > 1 ? languages.map(code => `<span class="language-chip">${escapeHtml(languageName(code))}<button type="button" data-remove-language="${escapeHtml(code)}" aria-label="Remover ${escapeHtml(languageName(code))}">×</button></span>`).join('') : '';
  $('#languageSelectionCount').textContent = `${languages.length} ${languages.length === 1 ? 'idioma selecionado' : 'idiomas selecionados'}`;
}
function renderLanguages() {
  const query = $('#languageSearch').value.trim().toLocaleLowerCase('pt-BR');
  const languages = state.catalog.languages.filter(language => `${language.name} ${language.code}`.toLocaleLowerCase('pt-BR').includes(query));
  $('#languageList').innerHTML = languages.map(language => {
    const compatible = language.clone_supported !== false;
    const validated = ['validated', 'approved', 'verified'].includes(language.validation_status);
    return `<button type="button" class="language-option ${state.languages.has(language.code) ? 'selected' : ''}" data-language="${escapeHtml(language.code)}" aria-pressed="${state.languages.has(language.code)}" ${compatible ? '' : 'disabled'}><span><strong>${escapeHtml(language.name)}</strong><small>${compatible ? validated ? 'Timbre original · validado' : 'Timbre original · em validação' : 'Timbre original indisponível'}</small></span><span>${state.languages.has(language.code) ? '✓' : '+'}</span></button>`;
  }).join('') || '<p class="dialog-note">Nenhum idioma encontrado.</p>';
  renderSelectedLanguages();
}
async function loadCatalog() {
  state.catalog = await api('/api/catalog');
  const available = state.catalog.languages.filter(language => language.clone_supported !== false);
  state.languages = new Set([...state.languages].filter(code => available.some(language => language.code === code)));
  if (!state.languages.size && available.length) state.languages.add(available.find(language => language.code.startsWith('pl'))?.code || available[0].code);
  $('#sourceLanguage').innerHTML = '<option value="auto">Detectar automaticamente</option>' + state.catalog.languages.map(language => `<option value="${escapeHtml(language.code)}">${escapeHtml(language.name)}</option>`).join('');
  renderLanguages();
}
async function checkHealth() {
  try {
    const data = await api('/api/health');
    state.health = data;
    $('#serverStatus').innerHTML = '<i></i>Estúdio conectado';
    $('#serverStatus').classList.remove('offline');
    $('#serverDot').classList.remove('offline');
    const validation = data.validation?.pipeline_validation || data.validation?.status || data.pipeline_validation;
    $('#validationNotice').hidden = ['complete', 'validated', 'approved', 'passed'].includes(validation);
  } catch {
    $('#serverStatus').innerHTML = '<i></i>Sem conexão';
    $('#serverStatus').classList.add('offline');
    $('#serverDot').classList.add('offline');
  }
}
async function loadSettings() {
  try {
    const payload = await api('/api/settings');
    state.settings = payload.config || payload;
    applySettingsToControls();
    if (state.jobs.length) renderJobs();
  } catch (error) { toast(error.message); }
}
function applySettingsToControls() {
  const studio = state.settings.studio || {};
  const quality = studio.performance_profile || studio.quality || 'fast_lipsync';
  $('#notebookUrl').value = state.settings.colab?.notebook_url || '';
  $('#defaultQuality').value = quality;
  $('#qualityProfile').value = quality;
  $('#driveCache').checked = studio.cache_on_drive !== false;
  $('#reviewMode').value = studio.review_mode || 'before_voice';
  $('#preserveAmbience').checked = studio.preserve_ambience !== false;
  updateNotebookLink();
  updateLipsyncOptions();
}
function updateNotebookLink() {
  const url = notebookUrl();
  $('#notebookLink').classList.toggle('disabled-link', !url);
  if (url) $('#notebookLink').href = url;
  else $('#notebookLink').removeAttribute('href');
  $('#notebookStatus').textContent = url ? 'Notebook conectado. Seu pacote será enviado manualmente pelo seletor do Colab.' : 'O notebook ainda não tem endereço público. Você pode baixar o .ipynb e abri-lo no Colab, ou salvar o endereço publicado aqui.';
}
async function saveSettings(event) {
  event.preventDefault();
  const button = $('button[type="submit"]', event.currentTarget);
  button.disabled = true;
  try {
    const result = await api('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ colab: { notebook_url: $('#notebookUrl').value.trim() }, studio: { performance_profile: $('#defaultQuality').value, cache_on_drive: $('#driveCache').checked } }) });
    state.settings = result.config || { ...state.settings, colab: { ...state.settings.colab, notebook_url: $('#notebookUrl').value.trim() }, studio: { ...state.settings.studio, performance_profile: $('#defaultQuality').value, cache_on_drive: $('#driveCache').checked } };
    applySettingsToControls();
    $('#settingsSaved').textContent = 'Configurações salvas.';
    if (state.jobs.length) renderJobs();
    toast('Seu estúdio está configurado.');
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}
function setVideo(file, useTransfer = false) {
  if (!file) return clearVideo();
  if (useTransfer) { const transfer = new DataTransfer(); transfer.items.add(file); $('#videoInput').files = transfer.files; }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  state.faceSelection = null;
  $('#faceBox').hidden = true;
  const preview = $('#sourcePreview');
  preview.src = state.previewUrl;
  $('#fileName').textContent = file.name;
  $('#fileMeta').textContent = `${formatBytes(file.size)} · lendo o vídeo…`;
  preview.onloadedmetadata = () => {
    $('#fileMeta').textContent = `${formatBytes(file.size)} · ${formatDuration(preview.duration)} · ${preview.videoWidth} × ${preview.videoHeight}`;
    if (preview.duration > 1200) toast('Este vídeo ultrapassa o limite de 20 minutos.');
  };
  preview.onerror = () => { $('#fileMeta').textContent = `${formatBytes(file.size)} · a prévia não é suportada pelo navegador`; };
  $('#dropZone').hidden = true;
  $('#fileCard').hidden = false;
}
function clearVideo() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  state.faceSelection = null;
  $('#faceBox').hidden = true;
  const preview = $('#sourcePreview');
  preview.pause(); preview.removeAttribute('src'); preview.load();
  $('#videoInput').value = '';
  $('#dropZone').hidden = false; $('#fileCard').hidden = true;
}
function faceGeometry() {
  const video = $('#sourcePreview');
  const bounds = $('#faceOverlay').getBoundingClientRect();
  const scale = Math.min(bounds.width / video.videoWidth, bounds.height / video.videoHeight);
  if (!Number.isFinite(scale) || scale <= 0) return null;
  const width = video.videoWidth * scale, height = video.videoHeight * scale;
  return { bounds, scale, offsetX: (bounds.width - width) / 2, offsetY: (bounds.height - height) / 2, width, height };
}
function facePoint(event) {
  const geometry = faceGeometry();
  if (!geometry) return null;
  const x = Math.max(0, Math.min(geometry.width, event.clientX - geometry.bounds.left - geometry.offsetX));
  const y = Math.max(0, Math.min(geometry.height, event.clientY - geometry.bounds.top - geometry.offsetY));
  return { x: Math.round(x / geometry.scale), y: Math.round(y / geometry.scale) };
}
function drawFaceSelection() {
  const geometry = faceGeometry();
  const selected = state.faceSelection;
  const box = $('#faceBox');
  if (!geometry || !selected) { box.hidden = true; return; }
  const [x1, y1, x2, y2] = selected.bbox;
  Object.assign(box.style, { left: `${geometry.offsetX + x1 * geometry.scale}px`, top: `${geometry.offsetY + y1 * geometry.scale}px`, width: `${(x2 - x1) * geometry.scale}px`, height: `${(y2 - y1) * geometry.scale}px` });
  box.hidden = false;
}
function updateFaceDrag(event) {
  const start = state.faceDrag;
  const end = facePoint(event);
  if (!start || !end) return;
  state.faceSelection = { bbox: [Math.min(start.x, end.x), Math.min(start.y, end.y), Math.max(start.x, end.x), Math.max(start.y, end.y)], frame_time: Number($('#sourcePreview').currentTime.toFixed(3)) };
  drawFaceSelection();
}
function downloadUrl(url) {
  const link = document.createElement('a');
  link.href = url; link.download = ''; document.body.appendChild(link); link.click(); link.remove();
}
async function startProject(event) {
  event.preventDefault();
  if (state.operation !== 'text_to_voice' && !$('#videoInput').files.length) return toast('Escolha seu vídeo para começar.');
  if (state.operation === 'lipsync_video' && !$('#audioInput').files.length) return toast('Escolha o áudio para sincronizar.');
  if (state.operation === 'text_to_voice' && !$('#referenceInput').files.length) return toast('Escolha uma referência vocal autorizada.');
  if (state.operation !== 'lipsync_video' && !state.languages.size) return toast('Escolha pelo menos um idioma.');
  if (!$('#voiceConsent').checked) return toast('Confirme a autorização das vozes e dos rostos.');
  if (usesLipsync() && $('#faceSelection').value === 'manual' && !state.faceSelection) return toast('Selecione o rosto arrastando um retângulo na prévia.');
  const button = $('#createButton');
  button.disabled = true;
  $('#createButtonLabel').textContent = 'Preparando seu projeto…';
  const url = notebookUrl();
  // Reserve the tab during the user gesture; asynchronous preparation may otherwise block the popup.
  let tab = url ? window.open('about:blank', '_blank') : null;
  if (tab) { tab.opener = null; tab.document.title = 'DubStudio — preparando projeto'; tab.document.body.textContent = 'Preparando seu pacote. O notebook abrirá em seguida.'; }
  const form = new FormData();
  const appendFile = (name, selector) => { const file = $(selector).files[0]; if (file) form.append(name, file); };
  if (state.operation !== 'text_to_voice') appendFile('video', '#videoInput');
  if (state.operation === 'lipsync_video') appendFile('audio', '#audioInput');
  if (state.operation === 'text_to_voice') appendFile('reference', '#referenceInput');
  form.append('operation', state.operation);
  form.append('target_languages', JSON.stringify(state.operation === 'lipsync_video' ? [] : [...state.languages]));
  form.append('source_language', $('#sourceLanguage').value);
  form.append('preset', 'original');
  form.append('performance_profile', $('#qualityProfile').value);
  form.append('review_mode', state.operation === 'dub_video' ? $('#reviewMode').value : 'automatic');
  form.append('require_lipsync', String(usesLipsync()));
  form.append('preserve_ambience', String(state.operation === 'dub_video' && $('#preserveAmbience').checked));
  form.append('cache_on_drive', String(state.settings.studio?.cache_on_drive !== false));
  form.append('voice_clone_consent', 'true');
  form.append('text', state.operation === 'text_to_voice' ? $('#voiceText').value.trim() : '');
  form.append('speaker_count', $('#speakerCount').value);
  form.append('subtitle_mode', state.operation === 'dub_video' ? $('#subtitleMode').value : 'none');
  const selectedFace = state.faceSelection;
  const videoWidth = $('#sourcePreview').videoWidth, videoHeight = $('#sourcePreview').videoHeight;
  form.append('face_selection', usesLipsync() && $('#faceSelection').value === 'manual' ? JSON.stringify({ bbox: [selectedFace.bbox[0] / videoWidth, selectedFace.bbox[1] / videoHeight, (selectedFace.bbox[2] - selectedFace.bbox[0]) / videoWidth, (selectedFace.bbox[3] - selectedFace.bbox[1]) / videoHeight], normalized: true, frame_time: selectedFace.frame_time }) : 'auto');
  try {
    const payload = await api('/api/jobs', { method: 'POST', body: form });
    const job = payload.job || payload;
    state.librarySection = 'projects';
    state.selectedJob = job.id;
    // Show the acknowledged project immediately, even if a pre-creation GET is
    // still in flight or the following list has not included it yet.
    state.jobs = [job, ...state.jobs.filter(item => item.id !== job.id)];
    state.launchPending = { id: job.id, job, tab, url, blocked: Boolean(url && !tab) };
    tab = null;
    renderJobs();
    showView('creations');
    await loadJobs({ quiet: true, invalidate: true });
    toast('Projeto criado. Preparando o pacote para o Colab.');
  } catch (error) { if (tab) tab.close(); toast(error.message); }
  finally { button.disabled = false; $('#createButtonLabel').textContent = 'Preparar e abrir no Colab'; }
}
function statusName(status) {
  if (status === 'legacy') return 'Histórico preservado';
  return ({ queued: 'Preparando', preparing: 'Preparando pacote', preparing_colab: 'Preparando pacote', awaiting_preparation: 'Seleção salva', running: 'Preparando pacote', awaiting_colab: 'Pacote pronto', awaiting_import: 'Aguardando resultado', processing_colab: 'No Colab', external_running: 'No Colab', review_required: 'Revisar tradução', awaiting_review: 'Revisar tradução', review_approved: 'Revisão aprovada', validating: 'Conferindo resultado', complete: 'Resultado pronto', partial: 'Resultados parciais', failed: 'Precisa de atenção', archived: 'Arquivado', cancelled: 'Interrompido' })[status] || 'Acompanhe pelo notebook';
}
function isExternal(job) { return ['awaiting_import', 'external_running', 'processing_colab'].includes(job.status); }
function jobTitle(job) { return job.options?.original_name || job.options?.title || job.metadata?.original_name || (job.options?.operation === 'text_to_voice' ? 'Narração de texto' : `Projeto ${job.id}`); }
function jobLanguages(job) { return job.options?.target_languages || (job.options?.language ? [job.options.language] : []); }
function outputsFor(job) {
  const value = job.metadata?.outputs || job.metadata?.studio_results || job.outputs || [];
  return Array.isArray(value) ? value : Object.entries(value).map(([language, output]) => ({ language, ...output }));
}
function outputFileKinds(output) {
  const files = output.files || {};
  return Array.isArray(files) ? files.map(file => typeof file === 'string' ? file : file.kind) : Object.keys(files);
}
function previewOutputs(job) {
  const isTts = job.options?.operation === 'text_to_voice';
  return outputsFor(job).filter(output => ['complete', 'approved', 'validated'].includes(output.status)).map(output => {
    const kinds = outputFileKinds(output);
    const kind = !isTts && kinds.includes('video') ? 'video' : kinds.includes('wav') ? 'wav' : kinds.includes('mp3') ? 'mp3' : null;
    return { language: output.language || output.code || 'original', kind };
  }).filter(output => output.kind);
}
function selectedPreview(job, versions = previewOutputs(job)) {
  const selected = versions.find(output => output.language === state.previewLanguages.get(job.id)) || versions[0];
  if (selected) state.previewLanguages.set(job.id, selected.language);
  return selected;
}
function renderJobPreview(job) {
  const versions = previewOutputs(job);
  const selected = selectedPreview(job, versions);
  if (!selected) return job.options?.operation === 'text_to_voice' ? '' : `<video controls playsinline preload="metadata" class="job-preview" src="${jobPath(job.id, '/source')}"></video>`;
  const selector = `<label class="field"><span>Idioma da prévia</span><select data-preview-language aria-label="Idioma da prévia">${versions.map(output => `<option value="${escapeHtml(output.language)}" ${output.language === selected.language ? 'selected' : ''}>${escapeHtml(languageName(output.language))}</option>`).join('')}</select></label>`;
  const source = outputPath(job, selected.language, selected.kind, true);
  const name = escapeHtml(`Prévia: ${languageName(selected.language)}`);
  const media = selected.kind === 'video' ? `<video controls playsinline preload="metadata" class="job-preview" data-result-preview aria-label="${name}" src="${source}"></video>` : `<audio controls preload="metadata" class="job-preview audio" data-result-preview aria-label="${name}" src="${source}"></audio>`;
  return selector + media;
}
function selectJobPreview(job, language) {
  const versions = previewOutputs(job);
  const chosen = versions.find(output => output.language === language);
  const detail = $('[data-current-job]', $('#jobDetail'));
  if (!chosen || !detail || detail.dataset.currentJob !== job.id) return false;
  let player = $('[data-result-preview]', detail);
  if (!player) return false;
  const selector = $('[data-preview-language]', detail);
  const source = outputPath(job, chosen.language, chosen.kind, true);
  state.previewLanguages.set(job.id, chosen.language);
  if (selector) selector.value = chosen.language;
  if (player.getAttribute('src') === source) return true;
  player.pause();
  const tag = chosen.kind === 'video' ? 'video' : 'audio';
  if (player.tagName.toLowerCase() !== tag) {
    const replacement = document.createElement(tag);
    replacement.setAttribute('controls', '');
    replacement.setAttribute('preload', 'metadata');
    replacement.setAttribute('data-result-preview', '');
    if (tag === 'video') replacement.setAttribute('playsinline', '');
    replacement.className = tag === 'video' ? 'job-preview' : 'job-preview audio';
    replacement.volume = player.volume;
    replacement.muted = player.muted;
    player.replaceWith(replacement);
    player = replacement;
  }
  player.setAttribute('aria-label', `Prévia: ${languageName(chosen.language)}`);
  player.setAttribute('src', source);
  player.load(); // Stream only this approved version; do not download or reload the page.
  return true;
}
function loadJobs({ quiet = false, invalidate = !quiet } = {}) {
  if (state.loadingJobs) {
    // Passive four-second ticks share this request. Only acknowledged user
    // changes invalidate a pre-change GET and need one coalesced follow-up.
    // Enqueuing every quiet tick can otherwise starve render and its callers.
    if (invalidate) state.jobsRefreshRequested = true;
    state.jobsRefreshQuiet = state.jobsRefreshQuiet && quiet;
    return state.jobsLoadPromise;
  }
  state.loadingJobs = true;
  state.jobsRefreshQuiet = quiet;
  state.jobsLoadPromise = (async () => {
    do {
      state.jobsRefreshRequested = false;
      try {
        const [payload, trash] = await Promise.all([api('/api/jobs'), api('/api/trash')]);
        // A mutating refresh during this GET belongs to newer user state.
        // Drain one coalesced follow-up before committing the old snapshot.
        if (state.jobsRefreshRequested) continue;
        state.jobs = (Array.isArray(payload) ? payload : payload.jobs || []).filter(job => job.status !== 'archived');
        const provisional = state.launchPending?.job;
        if (provisional) {
          if (state.jobs.some(job => job.id === provisional.id)) state.launchPending.job = null;
          else state.jobs.unshift(provisional);
        }
        state.trashJobs = trash.jobs || [];
        $('#jobCount').textContent = state.jobs.length;
        $('#lastRefresh').textContent = 'Atualizado agora';
        renderJobs();
        await finishLaunchIfReady();
      } catch (error) { if (!state.jobsRefreshRequested && !state.jobsRefreshQuiet) toast(error.message); }
    } while (state.jobsRefreshRequested);
  })().finally(() => {
    state.loadingJobs = false;
    state.jobsLoadPromise = null;
  });
  return state.jobsLoadPromise;
}
function restoreLibrarySelection() {
  if (state.selectedJob || state.librarySection !== 'projects') return;
  try {
    const selection = JSON.parse(sessionStorage.getItem('dubstudio.librarySelection') || 'null');
    if (!selection || !['projects', 'trash'].includes(selection.section)
        || (selection.jobId !== null && (typeof selection.jobId !== 'string' || !selection.jobId || selection.jobId.length > 128))) return;
    state.librarySection = selection.section;
    state.selectedJob = selection.jobId;
  } catch (_) { /* Storage can be unavailable in private or restricted tabs. */ }
}
function rememberLibrarySelection() {
  try {
    sessionStorage.setItem('dubstudio.librarySelection', JSON.stringify({ section: state.librarySection, jobId: state.selectedJob }));
  } catch (_) { /* Selection still works without browser persistence. */ }
}
function visibleJobs() {
  return state.librarySection === 'trash' ? state.trashJobs : state.jobs;
}
function installLibraryFilter() {
  const header = $('.library-header');
  if (!header || $('[data-library-section]')) return;
  header.insertAdjacentHTML('afterend', '<div class="library-tabs" role="group" aria-label="Exibir projetos"><button type="button" data-library-section="projects" aria-pressed="true">Projetos</button><button type="button" data-library-section="trash" aria-pressed="false"><span data-icon="trash"></span>Lixeira</button></div>');
  paintIcons(header.parentElement);
  $$('[data-library-section]').forEach(button => button.addEventListener('click', () => {
    state.librarySection = button.dataset.librarySection;
    state.selectedJob = null;
    renderJobs();
    loadJobs({ quiet: true });
  }));
}
function renderJobs() {
  const jobs = visibleJobs();
  $$('[data-library-section]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.librarySection === state.librarySection)));
  if (!state.selectedJob || !jobs.some(job => job.id === state.selectedJob)) state.selectedJob = jobs[0]?.id || null;
  rememberLibrarySelection();
  if (state.review && (state.review.jobId !== state.selectedJob || state.librarySection !== 'projects')) state.review = null;
  const jobList = $('#jobList');
  const markup = jobs.map(job => `<button class="job-item ${job.id === state.selectedJob ? 'active' : ''}" data-job="${escapeHtml(job.id)}"><span class="job-item-title">${escapeHtml(jobTitle(job))}</span><span class="job-item-meta"><i class="${job.status === 'complete' ? 'ready' : job.status === 'failed' ? 'failed' : ''}"></i>${job.status === 'deleted' ? 'Na lixeira' : escapeHtml(statusName(job.status))}</span><span class="job-item-languages">${escapeHtml(jobLanguages(job).map(languageName).join(' · ') || 'Lip-sync com áudio')}</span></button>`).join('') || (state.librarySection === 'trash' ? '<div class="empty-state"><span data-icon="trash"></span><h3>Lixeira vazia</h3><p>Os projetos excluídos ficam aqui e podem ser restaurados.</p></div>' : '<div class="empty-state"><span data-icon="folder"></span><h3>Seu espaço está pronto</h3><p>As suas criações aparecerão aqui.</p><button class="primary" data-new-project>Criar um projeto</button></div>');
  // Unchanged polling must not replace focused/clicked project buttons.
  if (jobList.dataset.renderSignature !== markup) {
    jobList.innerHTML = markup;
    jobList.dataset.renderSignature = markup;
    paintIcons(jobList);
  }
  const selected = jobs.find(job => job.id === state.selectedJob);
  if (selected) renderJobDetail(selected);
  else $('#jobDetail').innerHTML = state.librarySection === 'trash' ? '<div class="empty-state"><span data-icon="trash"></span><h3>Seus arquivos continuam preservados</h3><p>Selecione um projeto da lixeira para restaurá-lo. Nenhum download ou arquivo externo ao Studio é apagado.</p></div>' : '<div class="empty-state"><span data-icon="spark"></span><h3>Sua próxima criação começa aqui</h3><p>Escolha um vídeo ou transforme suas palavras em voz.</p><button class="primary" data-new-project>Começar um projeto</button></div>';
  paintIcons($('#jobDetail'));
}
function detailSignature(job) {
  return JSON.stringify([job.id, job.status, job.has_colab_package, job.has_output, job.has_report, job.error, outputsFor(job), Boolean(job.metadata?.review), job.options?.operation, job.options?.require_lipsync, job.options?.face_selection, Boolean(job.metadata?.face_selection_preparation_required), notebookUrl()]);
}
function renderOutput(job, output) {
  const language = output.language || output.code || 'original';
  const status = output.status || 'pending';
  const approved = ['complete', 'approved', 'validated'].includes(status);
  const fileKinds = outputFileKinds(output);
  const names = { video: 'Baixar vídeo', wav: 'Áudio WAV', mp3: 'Áudio MP3', report: 'Relatório', original_srt: 'SRT original', translation_srt: 'SRT traduzido' };
  const links = approved ? fileKinds.filter(kind => names[kind]).map(kind => `<a href="${outputPath(job, language, kind)}" download class="${kind === 'video' || kind === 'wav' ? 'primary' : ''}">${names[kind]}</a>`).join('') : '';
  const isVideo = job.options?.operation !== 'text_to_voice';
  const confirmed = output.validation?.lipsync_confirmed === true || output.lipsync?.confirmed === true || output.report?.lipsync?.confirmed === true;
  // The report proves application to confirmed face windows, not perceptual quality.
  const lipBadge = isVideo ? job.options?.require_lipsync === false ? '<span class="status-badge">Sem lip-sync</span>' : approved && confirmed ? '<span class="status-badge ready">Lip-sync aplicado</span>' : '<span class="status-badge">Lip-sync não confirmado</span>' : '';
  return `<section class="output-card"><div class="output-heading"><span data-icon="globe"></span><strong>${escapeHtml(languageName(language))}</strong><span class="status-badge ${approved ? 'ready' : status === 'failed' ? 'failed' : ''}">${approved ? 'Arquivos conferidos' : status === 'failed' ? 'Falhou' : 'Em validação'}</span>${lipBadge}</div>${links ? `<div class="output-files">${links}</div>` : ''}<p class="output-note">${escapeHtml(output.error || output.message || (approved ? 'Arquivos conferidos. Reproduza esta versão para conferir a qualidade.' : 'O download será liberado quando a importação conferir os arquivos desta versão.'))}</p></section>`;
}
function renderJobDetail(job) {
  if (job.status === 'deleted') {
    $('#jobDetail').innerHTML = `<div class="job-detail-content"><div class="job-kicker"><div><span class="eyebrow">PROJETO ${escapeHtml(job.id.toUpperCase())}</span><h2>${escapeHtml(jobTitle(job))}</h2></div><span class="status-badge">Na lixeira</span></div><p class="job-message">A cópia local deste projeto está preservada na lixeira. Restaurar devolve o projeto e seus arquivos à biblioteca, sem iniciar ou cancelar nada no Colab.</p><p class="settings-help">Downloads e arquivos originais fora do Studio continuam intactos.</p><div class="job-actions"><span class="quiet-label">Excluído em ${new Date(Number(job.deleted_at || 0) * 1000).toLocaleDateString('pt-BR')}</span><button class="secondary restore-button" type="button" data-action="restore">Restaurar projeto</button></div></div>`;
    return;
  }
  const legacy = job.status === 'legacy' || job.execution_location === 'legacy' || job.metadata?.legacy === true;
  const existing = $('[data-current-job]', $('#jobDetail'));
  const signature = detailSignature(job);
  if (existing && existing.dataset.currentJob === job.id && existing.dataset.signature === signature) {
    $('[data-job-message]', existing).textContent = job.message || job.stage || '';
    return; // Keep the video element intact during polling so playback never resets.
  }
  const outputs = outputsFor(job);
  const preview = renderJobPreview(job);
  if (legacy) {
    $('#jobDetail').innerHTML = `<div class="job-detail-content" data-current-job="${escapeHtml(job.id)}"><div class="job-kicker"><div><span class="eyebrow">PROJETO ${escapeHtml(job.id.toUpperCase())}</span><h2>${escapeHtml(jobTitle(job))}</h2></div><span class="status-badge">Histórico preservado</span></div><p class="job-message" data-job-message>${escapeHtml(job.message || 'Arquivos anteriores preservados.')}</p>${preview}<p class="settings-help">Este projeto usa o formato antigo. Para processá-lo no Colab, crie um novo projeto em Dublar vídeo. Nenhuma tarefa antiga será retomada automaticamente.</p><div class="job-actions"><span class="quiet-label">${new Date(Number(job.created_at || 0) * 1000).toLocaleDateString('pt-BR')}</span><button class="archive-button" type="button" data-action="archive">Arquivar projeto</button><button class="delete-button" type="button" data-action="trash"><span data-icon="trash"></span>Excluir projeto</button></div></div>`;
    const legacyDetail = $('[data-current-job]', $('#jobDetail'));
    legacyDetail.dataset.signature = signature;
    paintIcons(legacyDetail);
    return;
  }
  const reviewAvailable = !legacy && (['review_required', 'awaiting_review'].includes(job.status) || job.metadata?.review || job.metadata?.review_available);
  const repairAvailable = canRepairFace(job);
  const url = notebookUrl();
  const workflow = job.has_colab_package ? `<section class="colab-workflow"><div class="workflow-head"><span class="colab-symbol small">co</span><h3>Continue sua criação no Colab</h3></div><p>Baixe o pacote, abra o notebook e escolha o ZIP no seletor de upload. Execute as células e importe aqui o resultado final.</p><div class="inline-actions"><button class="primary" type="button" data-action="open-colab"><span data-icon="external"></span>Abrir no Colab</button><a href="${jobPath(job.id, '/colab-package')}" class="secondary" download><span data-icon="download"></span>Baixar pacote</a></div>${url ? '' : '<p class="settings-help">Salve o endereço público em Configurações, ou baixe o notebook .ipynb para abri-lo manualmente.</p><a class="text-link" href="/api/colab/notebook" download>Baixar notebook .ipynb</a>'}<label class="import-drop"><span data-icon="upload"></span><span>Importar resultado final (.zip)</span><input class="visually-hidden" type="file" data-import="result" accept=".zip,application/zip"></label><label class="import-drop"><span data-icon="translate"></span><span>Importar tradução para revisão (.zip)</span><input class="visually-hidden" type="file" data-import="review" accept=".zip,application/zip"></label></section>` : '<div class="external-progress"><span data-icon="folder"></span>Preparando seu pacote neste computador…</div>';
  $('#jobDetail').innerHTML = `<div class="job-detail-content" data-current-job="${escapeHtml(job.id)}"><div class="job-kicker"><div><span class="eyebrow">PROJETO ${escapeHtml(job.id.toUpperCase())}</span><h2>${escapeHtml(jobTitle(job))}</h2></div><span class="status-badge ${job.status === 'complete' ? 'ready' : job.status === 'failed' ? 'failed' : ''}">${escapeHtml(statusName(job.status))}</span></div><p class="job-message" data-job-message>${escapeHtml(job.message || job.stage || '')}</p>${isExternal(job) ? '<div class="external-progress"><span data-icon="colab"></span>Execução no Colab — acompanhe pelo notebook</div>' : ''}${preview}${job.error ? `<div class="error-banner">${escapeHtml(job.error)}</div>` : ''}${repairAvailable ? '<section class="review-card"><h3>Corrigir confirmação facial</h3><p>Marque o rosto no instante indicado pela falha. O notebook ainda verificará a trilha; a seleção não garante aprovação automática.</p><button class="secondary" type="button" data-action="repair-face">Selecionar rosto para retomada</button></section>' : ''}${job.status === 'awaiting_preparation' ? '<section class="review-card"><h3>Seleção salva</h3><p>Prepare um novo pacote corrigido. O pacote anterior permanece no histórico; isso não executa nada no Colab.</p><button class="primary" type="button" data-action="prepare-corrected">Preparar pacote corrigido</button></section>' : workflow}${reviewAvailable ? '<section class="review-card"><h3>Revisão da tradução</h3><p>Confira e ajuste as falas antes de sintetizar as vozes. Os tempos originais serão mantidos.</p><div class="inline-actions"><button class="secondary" type="button" data-action="load-review">Abrir revisão</button></div><div id="reviewEditor"></div></section>' : ''}${outputs.length ? `<div class="result-header"><h3>Versões do seu projeto</h3><span class="quiet-label">${outputs.length} ${outputs.length === 1 ? 'idioma' : 'idiomas'}</span></div>${outputs.map(output => renderOutput(job, output)).join('')}` : ''}<div class="job-actions"><span class="quiet-label">${new Date(Number(job.created_at || 0) * 1000).toLocaleDateString('pt-BR')}</span><button class="archive-button" type="button" data-action="archive">Arquivar projeto</button><button class="delete-button" type="button" data-action="trash"><span data-icon="trash"></span>Excluir projeto</button></div></div>`;
  const detail = $('[data-current-job]', $('#jobDetail'));
  detail.dataset.signature = signature;
  // Keep the chooser bound to the project that created it, even if polling or
  // another selection replaces the visible detail before its change event.
  $$('[data-import]', detail).forEach(input => { input.dataset.importJob = job.id; });
  paintIcons(detail);
}
async function finishLaunchIfReady() {
  const pending = state.launchPending;
  if (!pending) return;
  const job = state.jobs.find(item => item.id === pending.id);
  if (!job) return;
  if (job.status === 'failed') {
    if (pending.tab && !pending.tab.closed) pending.tab.close();
    state.launchPending = null;
    return;
  }
  if (!job.has_colab_package) return;
  state.launchPending = null;
  downloadUrl(jobPath(job.id, '/colab-package'));
  if (pending.tab && !pending.tab.closed && pending.url) {
    pending.tab.location.replace(pending.url);
    await api(jobPath(job.id, '/external-started'), { method: 'POST' }).catch(() => {});
    toast('Pacote baixado. No Colab, execute as células e envie o ZIP.');
  } else if (pending.url) toast('O pacote está pronto. Se a aba não abriu, use “Abrir no Colab” e “Baixar pacote”.');
  else toast('Pacote pronto. Configure o endereço público ou baixe o notebook para abri-lo no Colab.');
}
async function openColab(job) {
  const url = notebookUrl();
  if (!url) { showView('settings'); toast('Salve o endereço público do notebook para abrir pelo Studio. O arquivo .ipynb também está disponível.'); return; }
  const tab = window.open(url, '_blank');
  if (tab) tab.opener = null;
  downloadUrl(jobPath(job.id, '/colab-package'));
  if (!tab) { toast('A aba foi bloqueada. Permita pop-ups ou abra o notebook pelo link em Configurações.'); return; }
  try { await api(jobPath(job.id, '/external-started'), { method: 'POST' }); await loadJobs({ quiet: true, invalidate: true }); }
  catch (error) { toast(error.message); }
}
async function importPackage(input) {
  const file = input.files[0];
  const jobId = input.dataset.importJob || input.closest?.('[data-current-job]')?.dataset.currentJob;
  const job = state.jobs.find(item => item.id === jobId);
  if (!file || !job) return;
  const review = input.dataset.import === 'review';
  const form = new FormData(); form.append(review ? 'review' : 'result', file);
  input.disabled = true;
  toast(review ? 'Importando a tradução para revisão…' : 'Conferindo o pacote e os arquivos de resultado…');
  try {
    await api(jobPath(job.id, review ? '/review' : '/colab-result'), { method: 'POST', body: form });
    await loadJobs({ quiet: true, invalidate: true });
    if (review && state.selectedJob === jobId && state.librarySection === 'projects') await loadReview(job);
    toast(review ? 'Tradução disponível para você revisar.' : 'Pacote recebido. Confira o estado de cada idioma.');
  } catch (error) { toast(error.message); }
  finally { input.disabled = false; input.value = ''; }
}
function reviewEditorForJob(jobId) {
  if (state.selectedJob !== jobId || state.librarySection !== 'projects') return null;
  const detail = $('[data-current-job]', $('#jobDetail'));
  return detail?.dataset.currentJob === jobId ? $('#reviewEditor', detail) : null;
}
async function loadReview(job) {
  const editor = reviewEditorForJob(job.id);
  if (!editor) return;
  const request = ++state.reviewRequest;
  try {
    const payload = await api(jobPath(job.id, '/review'));
    // Reject responses for replaced editors (including A -> B -> A) and older
    // requests for the same editor before mutating review state or the DOM.
    if (request !== state.reviewRequest || reviewEditorForJob(job.id) !== editor) return;
    const review = payload.review || payload;
    const translations = review.translations || {};
    state.review = { jobId: job.id, payload: structuredClone(review), translations };
    editor.innerHTML = `<div class="review-scroll">${Object.entries(translations).map(([language, segments]) => `<h4 class="review-language">${escapeHtml(languageName(language))}</h4>${(Array.isArray(segments) ? segments : segments.segments || []).map((segment, index) => `<div class="review-segment"><small>${formatDuration(segment.start)} → ${formatDuration(segment.end)} · ${escapeHtml(segment.source_text || segment.original || segment.text || '')}</small><textarea data-review-language="${escapeHtml(language)}" data-review-index="${index}" aria-label="Tradução ${escapeHtml(languageName(language))}, trecho ${index + 1}">${escapeHtml(segment.translated_text || segment.translation || '')}</textarea></div>`).join('')}`).join('')}</div><div class="inline-actions"><button class="primary" type="button" data-action="save-review">Aprovar e baixar revisão</button></div>`;
    if (!Object.keys(translations).length) editor.innerHTML = '<p class="settings-help">Nenhuma tradução disponível neste pacote.</p>';
  } catch (error) { if (request === state.reviewRequest && reviewEditorForJob(job.id) === editor) toast(error.message); }
}
async function saveReview(job, button) {
  const editor = reviewEditorForJob(job.id);
  if (!editor || !state.review || state.review.jobId !== job.id) return;
  button.disabled = true;
  const translations = structuredClone(state.review.translations);
  $$('[data-review-language]', editor).forEach(input => {
    const value = translations[input.dataset.reviewLanguage];
    const segments = Array.isArray(value) ? value : value.segments;
    const segment = segments[Number(input.dataset.reviewIndex)];
    segment.translated_text = input.value.trim();
  });
  try {
    const payload = await api(jobPath(job.id, '/review-approved'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...state.review.payload, translations }) });
    downloadUrl(payload.download_url || payload.url || payload.package_url || jobPath(job.id, '/review-approved'));
    toast('Revisão aprovada. Envie este ZIP no notebook para retomar a geração das vozes.');
    await loadJobs({ quiet: true, invalidate: true });
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}
function canRepairFace(job) {
  return ['failed', 'awaiting_colab', 'awaiting_import', 'review_required', 'review_approved', 'awaiting_preparation'].includes(job.status)
    && ['dub_video', 'lipsync_video'].includes(job.options?.operation) && job.options?.require_lipsync === true
    && !outputsFor(job).some(output => ['complete', 'approved', 'validated'].includes(output.status));
}
function faceRepairSelection(values, duration, width, height) {
  const [time, x, y, w, h] = values.map(Number);
  if (![time,x,y,w,h,duration,width,height].every(Number.isFinite) || time < 0 || time >= duration
      || x < 0 || y < 0 || w <= 0 || h <= 0 || x+w > 1 || y+h > 1 || w*width < 20 || h*height < 20) return null;
  return {bbox:[x,y,w,h], normalized:true, frame_time:time};
}
function repairFace(job) {
  if (!canRepairFace(job) || $('#faceRepairDialog')) return;
  const dialog = document.createElement('dialog');
  dialog.id = 'faceRepairDialog'; dialog.className = 'project-action-dialog face-repair-dialog';
  dialog.setAttribute('aria-labelledby', 'faceRepairTitle');
  dialog.innerHTML = `<h2 id="faceRepairTitle">Selecionar rosto para retomada</h2><p>Informe o instante da cena, depois arraste um retângulo sobre o rosto falante. A seleção vale somente para a trilha detectada nesse instante; outros rostos continuam sendo verificados.</p><label>Instante da cena (segundos)<input id="repairTime" type="number" min="0" step="0.01" value="0"></label><div class="face-repair-video-wrap"><video id="repairVideo" muted preload="metadata" src="${jobPath(job.id, '/source')}"></video><div id="repairOverlay" class="face-repair-overlay" aria-label="Arraste para selecionar o rosto"><div id="repairBox" hidden></div></div></div><details><summary>Coordenadas normalizadas da seleção</summary><div class="face-repair-coordinates">${['X','Y','Largura','Altura'].map((label,i)=>`<label>${label}<input data-repair-coordinate="${i}" aria-label="${label} da seleção facial" type="number" min="0" max="1" step="0.001" value="0"></label>`).join('')}</div></details><p id="repairError" role="status"></p><div class="project-dialog-actions"><button class="secondary" type="button" data-cancel-repair>Cancelar</button><button class="primary" type="button" data-save-repair>Salvar seleção facial</button></div>`;
  document.body.append(dialog);
  const video = $('#repairVideo',dialog), overlay = $('#repairOverlay',dialog), box = $('#repairBox',dialog), time = $('#repairTime',dialog);
  const coordinates = $$('[data-repair-coordinate]',dialog);
  const previousFocus = document.activeElement;
  let start = null;
  const close = () => { video.pause(); if (dialog.open) dialog.close(); dialog.remove(); previousFocus?.focus?.(); };
  const geometry = () => { const bounds=overlay.getBoundingClientRect(), scale=Math.min(bounds.width/video.videoWidth,bounds.height/video.videoHeight); return {bounds,scale,ox:(bounds.width-video.videoWidth*scale)/2,oy:(bounds.height-video.videoHeight*scale)/2}; };
  const point = event => { const g=geometry(); if (!Number.isFinite(g.scale) || g.scale<=0) return null; return {x:Math.max(0,Math.min(1,(event.clientX-g.bounds.left-g.ox)/(video.videoWidth*g.scale))),y:Math.max(0,Math.min(1,(event.clientY-g.bounds.top-g.oy)/(video.videoHeight*g.scale)))}; };
  const draw = () => { const [x,y,w,h]=coordinates.map(input=>Number(input.value)),g=geometry(); if (![x,y,w,h,g.scale].every(Number.isFinite)||g.scale<=0||w<=0||h<=0) {box.hidden=true;return;} Object.assign(box.style,{left:`${g.ox+x*video.videoWidth*g.scale}px`,top:`${g.oy+y*video.videoHeight*g.scale}px`,width:`${w*video.videoWidth*g.scale}px`,height:`${h*video.videoHeight*g.scale}px`});box.hidden=false; };
  const seekSelection = () => { const t=Number(time.value); if(Number.isFinite(video.duration)&&t>=0&&t<video.duration)video.currentTime=t; };
  time.addEventListener('input',seekSelection); time.addEventListener('change',seekSelection);
  video.addEventListener('loadedmetadata',()=>{const current=job.options?.face_selection;if(current&&typeof current==='object'){time.value=current.frame_time;coordinates.forEach((input,i)=>input.value=current.bbox[i]);}seekSelection();draw();});
  coordinates.forEach(input=>input.addEventListener('input',draw));
  overlay.addEventListener('pointerdown',event=>{ if(event.button!==0)return; start=point(event); if(start){event.preventDefault();overlay.setPointerCapture(event.pointerId);} });
  const move = event => { const end=point(event);if(!start||!end)return;const values=[Math.min(start.x,end.x),Math.min(start.y,end.y),Math.abs(end.x-start.x),Math.abs(end.y-start.y)];coordinates.forEach((input,i)=>input.value=values[i].toFixed(6));draw(); };
  overlay.addEventListener('pointermove',move);overlay.addEventListener('pointerup',event=>{move(event);start=null;});overlay.addEventListener('pointercancel',()=>start=null);
  $('[data-cancel-repair]',dialog).addEventListener('click',close);
  dialog.addEventListener('cancel',event=>{event.preventDefault();close();});
  $('[data-save-repair]',dialog).addEventListener('click',async event=>{const selection=faceRepairSelection([time.value,...coordinates.map(input=>input.value)],video.duration,video.videoWidth,video.videoHeight);if(!selection){$('#repairError',dialog).textContent='Marque uma área válida do rosto e um instante dentro do vídeo.';return;}event.currentTarget.disabled=true;try{await api(jobPath(job.id,'/face-selection'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({face_selection:selection})});close();await loadJobs();toast('Seleção salva. Prepare o pacote corrigido para enviar ao Colab.');}catch(error){$('#repairError',dialog).textContent=error.message;event.currentTarget.disabled=false;}});
  dialog.showModal(); $('[data-cancel-repair]',dialog).focus();
}
function confirmProjectAction(job, restoring = false) {
  if ($('#projectActionDialog')) return Promise.resolve(false);
  return new Promise(resolve => {
    const previousFocus = document.activeElement;
    const dialog = document.createElement('dialog');
    dialog.id = 'projectActionDialog';
    dialog.className = 'project-action-dialog';
    dialog.setAttribute('aria-labelledby', 'projectActionTitle');
    dialog.setAttribute('aria-describedby', 'projectActionDescription');
    const heading = document.createElement('h2');
    heading.id = 'projectActionTitle';
    heading.textContent = restoring ? 'Restaurar projeto?' : 'Excluir projeto?';
    const description = document.createElement('p');
    description.id = 'projectActionDescription';
    description.textContent = restoring
      ? `Restaurar “${jobTitle(job)}” à biblioteca? Seus arquivos voltam a ficar disponíveis. Isso não inicia a execução no Colab.`
      : `Excluir “${jobTitle(job)}” da biblioteca? A cópia local será movida para a Lixeira e poderá ser restaurada. Downloads e arquivos originais fora do Studio ficam intactos.`;
    dialog.append(heading, description);
    if (!restoring && isExternal(job)) {
      const warning = document.createElement('p');
      warning.className = 'project-action-warning';
      warning.textContent = 'Atenção: isso NÃO cancela a execução no Colab. Acompanhe ou interrompa o processamento no próprio notebook.';
      dialog.append(warning);
      warning.id = 'projectActionWarning';
      dialog.setAttribute('aria-describedby', 'projectActionDescription projectActionWarning');
    }
    const actions = document.createElement('div');
    actions.className = 'project-dialog-actions';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'secondary';
    cancel.textContent = 'Cancelar';
    const confirm = document.createElement('button');
    confirm.type = 'button';
    confirm.className = restoring ? 'primary' : 'delete-button';
    confirm.textContent = restoring ? 'Restaurar projeto' : 'Excluir projeto';
    actions.append(cancel, confirm);
    dialog.append(actions);
    let settled = false;
    function finish(accepted) {
      if (settled) return;
      settled = true;
      if (dialog.open) dialog.close();
      dialog.remove();
      previousFocus?.focus?.();
      resolve(accepted);
    }
    cancel.addEventListener('click', () => finish(false));
    confirm.addEventListener('click', () => finish(true));
    dialog.addEventListener('cancel', event => { event.preventDefault(); finish(false); });
    dialog.addEventListener('close', () => finish(false));
    document.body.append(dialog);
    try {
      dialog.showModal();
      cancel.focus(); // Cancellation is always the safe default keyboard action.
    } catch (_) { finish(false); }
  });
}
async function jobAction(event) {
  const button = event.target.closest('[data-action]');
  if (!button) return;
  const job = visibleJobs().find(item => item.id === state.selectedJob);
  if (!job) return;
  if (button.dataset.action === 'trash' || button.dataset.action === 'restore') {
    const restoring = button.dataset.action === 'restore';
    if (!await confirmProjectAction(job, restoring)) return;
    button.disabled = true;
    try {
      await api(restoring ? `/api/trash/${encodeURIComponent(job.id)}/restore` : jobPath(job.id, '/trash'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirm: true }) });
      if (restoring) state.librarySection = 'projects';
      if (!restoring && state.launchPending?.id === job.id) state.launchPending = null;
      state.selectedJob = restoring ? job.id : null;
      state.review = null;
      await loadJobs();
      toast(restoring ? 'Projeto restaurado. Seus arquivos estão disponíveis novamente.' : 'Projeto movido para a Lixeira. Você pode restaurá-lo quando quiser.');
    } catch (error) { button.disabled = false; toast(error.message); }
    return;
  }
  if (button.dataset.action === 'open-colab') return openColab(job);
  if (button.dataset.action === 'repair-face') return repairFace(job);
  if (button.dataset.action === 'prepare-corrected') {
    button.disabled = true;
    try { await api(jobPath(job.id, '/prepare-colab'), {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true})});await loadJobs();toast('Preparando o pacote corrigido neste computador. Nenhuma GPU foi iniciada.'); }
    catch(error) {button.disabled=false;toast(error.message);}
    return;
  }
  if (button.dataset.action === 'load-review') return loadReview(job);
  if (button.dataset.action === 'save-review') return saveReview(job, button);
  if (button.dataset.action === 'archive') {
    if (!window.confirm('Arquivar este projeto? Os arquivos serão preservados.')) return;
    button.disabled = true;
    try { await api(jobPath(job.id), { method: 'DELETE' }); if (state.launchPending?.id === job.id) state.launchPending = null; state.selectedJob = null; await loadJobs(); toast('Projeto arquivado. Os arquivos foram preservados.'); }
    catch (error) { button.disabled = false; toast(error.message); }
  }
}
function bindEvents() {
  installLibraryFilter();
  $$('.nav-item').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
  $('.brand').addEventListener('click', event => { event.preventDefault(); showView('dub'); });
  window.addEventListener('hashchange', () => showView(location.hash.slice(1), { updateHash: false }));
  $('#projectForm').addEventListener('submit', startProject);
  $('#settingsForm').addEventListener('submit', saveSettings);
  $('#videoInput').addEventListener('change', event => setVideo(event.target.files[0]));
  $('#removeFile').addEventListener('click', clearVideo);
  $('#requireLipsync').addEventListener('change', event => { state.dubLipsync = event.target.checked; updateLipsyncOptions(); });
  $('#faceSelection').addEventListener('change', event => {
    const manual = event.target.value === 'manual';
    $('#faceOverlay').hidden = !manual;
    $('#faceHint').textContent = manual ? 'Pause o vídeo no rosto desejado e desenhe um retângulo na prévia.' : 'Se houver dúvida, o notebook pedirá confirmação.';
    if (manual) $('#sourcePreview').pause();
  });
  $('#faceOverlay').addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    const point = facePoint(event);
    if (!point) return;
    event.preventDefault();
    state.faceDrag = point;
    event.currentTarget.setPointerCapture(event.pointerId);
    updateFaceDrag(event);
  });
  $('#faceOverlay').addEventListener('pointermove', updateFaceDrag);
  $('#faceOverlay').addEventListener('pointerup', event => {
    updateFaceDrag(event); state.faceDrag = null;
    if (state.faceSelection) {
      const [x1, y1, x2, y2] = state.faceSelection.bbox;
      if (x2 - x1 < 20 || y2 - y1 < 20) { state.faceSelection = null; $('#faceBox').hidden = true; toast('Selecione uma área maior ao redor do rosto.'); }
      else $('#faceHint').textContent = 'Rosto marcado na prévia. O notebook ainda confirmará o falante.';
    }
  });
  $('#faceOverlay').addEventListener('pointercancel', () => { state.faceDrag = null; });
  window.addEventListener('resize', drawFaceSelection);
  $('#dropZone').addEventListener('dragover', event => { event.preventDefault(); event.currentTarget.classList.add('drag-over'); });
  $('#dropZone').addEventListener('dragleave', event => event.currentTarget.classList.remove('drag-over'));
  $('#dropZone').addEventListener('drop', event => { event.preventDefault(); event.currentTarget.classList.remove('drag-over'); if (event.dataTransfer.files[0]) setVideo(event.dataTransfer.files[0], true); });
  $('#audioInput').addEventListener('change', event => { $('#audioName').textContent = event.target.files[0]?.name || 'Escolher áudio'; });
  $('#referenceInput').addEventListener('change', event => { $('#referenceName').textContent = event.target.files[0]?.name || 'Escolher referência vocal'; });
  $('#voiceText').addEventListener('input', event => { $('#textCount').textContent = `${event.target.value.length.toLocaleString('pt-BR')} / 12.000`; });
  $('#advancedToggle').addEventListener('click', event => { const button = event.currentTarget; const expanded = button.getAttribute('aria-expanded') === 'true'; button.setAttribute('aria-expanded', String(!expanded)); $('#advancedFields').hidden = expanded; });
  $('#languageButton').addEventListener('click', () => { $('#languageSearch').value = ''; renderLanguages(); $('#languageDialog').showModal(); $('#languageSearch').focus(); });
  $('#closeLanguages').addEventListener('click', () => $('#languageDialog').close());
  $('#confirmLanguages').addEventListener('click', () => { if (!state.languages.size) return toast('Escolha pelo menos um idioma.'); $('#languageDialog').close(); renderSelectedLanguages(); });
  $('#languageSearch').addEventListener('input', renderLanguages);
  $('#languageList').addEventListener('click', event => { const button = event.target.closest('[data-language]'); if (!button || button.disabled) return; const code = button.dataset.language; if (state.languages.has(code)) state.languages.delete(code); else state.languages.add(code); renderLanguages(); });
  $('#selectedLanguages').addEventListener('click', event => { const button = event.target.closest('[data-remove-language]'); if (button) { state.languages.delete(button.dataset.removeLanguage); renderSelectedLanguages(); } });
  $('#refreshJobs').addEventListener('click', async event => { const button = event.currentTarget; button.disabled = true; await Promise.all([loadJobs(), checkHealth()]); button.disabled = false; });
  $('#jobList').addEventListener('click', event => { const button = event.target.closest('[data-job]'); if (!button) return; state.selectedJob = button.dataset.job; renderJobs(); });
  $('#jobDetail').addEventListener('click', jobAction);
  $('#jobDetail').addEventListener('change', event => {
    if (event.target.matches('[data-preview-language]')) {
      const job = state.jobs.find(item => item.id === state.selectedJob);
      if (job) selectJobPreview(job, event.target.value);
    } else if (event.target.matches('[data-import]')) importPackage(event.target);
  });
  document.addEventListener('click', event => { if (event.target.closest('[data-new-project]')) showView('dub'); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { loadJobs({ quiet: true }); checkHealth(); } });
}
async function init() {
  restoreLibrarySelection();
  paintIcons(); bindEvents(); showView(location.hash.slice(1) || 'dub', { updateHash: false });
  const results = await Promise.allSettled([loadCatalog(), loadSettings(), checkHealth(), loadJobs({ quiet: true })]);
  results.forEach(result => { if (result.status === 'rejected') toast(result.reason.message); });
  setInterval(() => { if (!document.hidden && (state.view === 'creations' || state.launchPending)) loadJobs({ quiet: true }); }, 4000);
  setInterval(() => { if (!document.hidden) checkHealth(); }, 30000);
}
init();
