// frontend/app.js
const BACKEND_BASE = "http://localhost:8000";

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const messagesContainer = document.getElementById("messages");
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");
  const newChatButton = document.getElementById("new-chat");
  const clearHistoryButton = document.getElementById("clear-history");
  const toggleThemeButton = document.getElementById("toggle-theme");
  const chatHistoryContainer = document.getElementById("chat-history");
  const currentChatTitle = document.getElementById("current-chat-title");
  const exportChatButton = document.getElementById("export-chat");
  const regenerateResponseButton = document.getElementById("regenerate-response");
  const stopResponseButton = document.getElementById("stop-response");
  const suggestionChips = document.querySelectorAll(".suggestion-chip");
  const fileUploadButton = document.getElementById("file-upload-button");
  const fileUploadInput = document.getElementById("file-upload");
  const voiceBtn = document.getElementById('voice-toggle');
  const voiceStatus = document.getElementById('voice-status');


  // --- Web Arama Toggle State ---
  let webSearchEnabled = JSON.parse(localStorage.getItem('webSearchEnabled') || 'false');
  const webToggleBtn = document.getElementById('web-toggle');

  function refreshWebToggleUI() {
    if (!webToggleBtn) return;
    webToggleBtn.classList.toggle('active', webSearchEnabled);
    webToggleBtn.setAttribute('aria-pressed', webSearchEnabled ? 'true' : 'false');
    webToggleBtn.title = webSearchEnabled ? 'Web Search: On' : 'Web Search: Off';
  }

  if (webToggleBtn) {
    refreshWebToggleUI();
    webToggleBtn.addEventListener('click', () => {
      webSearchEnabled = !webSearchEnabled;
      localStorage.setItem('webSearchEnabled', JSON.stringify(webSearchEnabled));
      refreshWebToggleUI();
    });
  }

  // State
  let currentChatId = null;
  let isTyping = false;
  let chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || {};
  let currentTheme = localStorage.getItem("theme") || "light";
  let typingSpeed = 2;
  let letterTimeout = null;
  let pendingFile = null;     // File objesi
  let stopGeneration = false; // (UI için, gerçek streaming yok)
  let mediaStream = null;
  let videoEl = null;

  async function openCamera() {
    if (mediaStream) return;
    const area = document.getElementById("camera-area");
    videoEl = document.getElementById("camera-preview");
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      videoEl.srcObject = mediaStream;
      area.style.display = "";
      area.removeAttribute("style");
      return "I will open the camera for you now.";
    } catch (e) {
      mediaStream = null;
      throw new Error("I can not open the camera: " + e.message);
    }
  }

  function closeCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    const area = document.getElementById("camera-area");
    if (area) area.style.display = "none";
    return "The camera was turned off.";
  }

  async function takePhotoBlob(mime = "image/jpeg", quality = 0.92) {
    if (!mediaStream || !videoEl) throw new Error("The camera is not on.");
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0);
    return await new Promise(res => canvas.toBlob(res, mime, quality));
  }

  // ---------- Backend API helpers ----------
  async function parseJsonResponse(response, endpointName) {
    let body = null;
    try {
      body = await response.json();
    } catch (err) {
      throw new Error(`${endpointName} returned invalid JSON.`);
    }

    if (!response.ok) {
      const detail = body && (body.detail || body.error || body.message);
      throw new Error(detail || `${endpointName} failed with status ${response.status}.`);
    }

    return body;
  }

  function apiRun(message, metadata = {}) {
    return fetch(`${BACKEND_BASE}/api/run`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ message, metadata }),
    }).then(r => parseJsonResponse(r, "/api/run"));
  }

  function apiIntent(text) {
    return fetch(`${BACKEND_BASE}/api/intent`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ text }),
    }).then(r => r.json());
  }
  function apiChat(message) {
    return fetch(`${BACKEND_BASE}/api/chat`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ message }),
    }).then(r => r.json());
  }
  function apiRag(question, useInternet = false, webOnly = false) {
    return fetch(`${BACKEND_BASE}/api/rag`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ question, use_internet: useInternet, web_only: webOnly }),
    }).then(r => r.json());
  }

  function apiDetectFromFile(file, draw=1) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("draw", String(draw));
    return fetch(`${BACKEND_BASE}/api/detect`, {
      method: "POST",
      body: fd
    }).then(r => r.json());
  }
  function apiPhotoFromFile(file) {
    const fd = new FormData();
    fd.append("file", file);
    return fetch(`${BACKEND_BASE}/api/photo`, {
      method: "POST",
      body: fd
    }).then(r => r.json());
  }

  function apiUploadDoc(file) {
    const fd = new FormData();
    fd.append("file", file);
    return fetch(`${BACKEND_BASE}/api/upload`, {
      method: "POST",
      body: fd
    }).then(r => r.json());
  }


  // ==== Voice Mode (English only) ====
  let voiceMode = false;
  let recognition = null;

  // Feature detection
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition || null;
  if (voiceBtn && !SR) {
    voiceBtn.disabled = true;
    voiceBtn.title = "Web Speech API not supported in this browser.";
  }

  function setVoiceStatus(state /* '','listening','thinking','speaking' */) {
    if (!voiceStatus) return;
    voiceStatus.className = '';
    if (state) voiceStatus.classList.add(state);
  }

  // TTS öncesi metni temizle: Resources kısmını ve çıplak URL satırlarını at
  function buildSpeakableText(fullText) {
    if (!fullText) return "";

    let text = String(fullText);

    // 1) "Resources:" ile başlayan kısmı tamamen kes
    const idx = text.indexOf("Resources:");
    if (idx !== -1) {
      text = text.slice(0, idx);
    }

    // 2) Satır satır gez: URL içeren satırları at
    const lines = text.split("\n");
    const filtered = lines.filter((line) => {
      const t = line.trim();
      if (!t) return false; // boş satırları at
      if (t.startsWith("http://") || t.startsWith("https://")) return false;
      if (t.startsWith("- http://") || t.startsWith("- https://")) return false;
      return true;
    });

    return filtered.join(" ").trim();
  }


  function speak(text) {
    if (!("speechSynthesis" in window)) return;

    const clean = buildSpeakableText(text);
    if (!clean) return;

    const u = new SpeechSynthesisUtterance(clean);

    const pickVoice = () => {
      const voices = window.speechSynthesis.getVoices() || [];
      const en = voices.find(v => /en-US|en_GB|en-/.test(v.lang)) || voices[0];
      if (en) u.voice = en;
    };

    pickVoice();
    window.speechSynthesis.onvoiceschanged = () => pickVoice();

    u.rate = 1.0;
    u.pitch = 1.0;

    setVoiceStatus('speaking');
    u.onend = () => setVoiceStatus('');

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }


  async function processVoiceTranscript(transcript) {
    try {
      const text = (transcript || "").trim();
      if (!text) return;

      // 0) Kullanıcı konuşmayı bitirdi → voice dinlemeyi KAPAT
      //    Böylece TTS sesi tekrar mikrofona girmez.
      if (typeof voiceMode !== "undefined" && voiceMode) {
        voiceMode = false;
        const btn = document.getElementById("voice-toggle");
        if (btn) btn.classList.remove("active");
      }
      if (typeof recognition !== "undefined" && recognition) {
        // onend içinde tekrar başlatmayı engelle
        recognition.onend = null;
        recognition.stop();
      }

			// 1) Kullanıcının söylediğini USER MESAJI olarak ekrana bas
			addMessageToUI("user", text);
			if (!chatHistory[currentChatId]) createNewChat();
			chatHistory[currentChatId].messages.push({ role: "user", content: text });
			saveChatHistory();

			// 2) Voice status → düşünme moduna al
			setVoiceStatus("thinking");

			// 3) Text mesajlarla aynı intent + komut + RAG pipeline'ına sok
			await runMessagePipeline(text, null);

			// 4) Pipeline içinde asistan cevabı UI'ya yazıldı; son mesajı seslendir
			const lastMsg = chatHistory[currentChatId].messages.slice(-1)[0];
			if (lastMsg && lastMsg.role === "assistant") {
				speak(lastMsg.content);
			}
		} catch (e) {
			console.error("voice error", e);
			speak("Sorry, there was an error.");
		} finally {
			setVoiceStatus("");
		}
	}


  function startRecognitionLoop() {
    if (!SR) return;
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    recognition.onstart = () => setVoiceStatus('listening');
    recognition.onerror = (e) => { console.warn('STT error', e); setVoiceStatus(''); };
    recognition.onend = () => {
      setVoiceStatus('');
      if (voiceMode) setTimeout(() => startRecognitionLoop(), 100);
    };
    recognition.onresult = (e) => {
      const transcript = (e.results?.[0]?.[0]?.transcript || "").trim();
      console.log("raw STT transcript:", transcript, e);
      processVoiceTranscript(transcript);
    };

    recognition.start();
  }

  if (voiceBtn) {
    voiceBtn.addEventListener('click', () => {
      if (!SR) return;
      voiceMode = !voiceMode;
      voiceBtn.classList.toggle('active', voiceMode);

      if (voiceMode) {
        window.speechSynthesis.cancel();
        startRecognitionLoop();
      } else {
        setVoiceStatus('');
        try { recognition && recognition.abort && recognition.abort(); } catch {}
        try { recognition && recognition.stop && recognition.stop(); } catch {}
      }
    });
  }

  // ---------- Init ----------
  init();

  function init() {
    // Theme — light is default
    if (currentTheme === "dark") {
      document.body.classList.add("dark-mode");
      toggleThemeButton.innerHTML = '<i class="fas fa-sun"></i><span>Light Mode</span>';
    } else {
      toggleThemeButton.innerHTML = '<i class="fas fa-moon"></i><span>Dark Mode</span>';
    }

    // Mobile sidebar toggle
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebar-overlay');
    if (mobileMenuBtn && sidebar) {
      mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        if (sidebarOverlay) sidebarOverlay.classList.toggle('visible');
      });
      if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', () => {
          sidebar.classList.remove('open');
          sidebarOverlay.classList.remove('visible');
        });
      }
    }

    // Events
    userInput.addEventListener("input", autoResizeTextarea);
    sendButton.addEventListener("click", handleSendMessage);
    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });
    newChatButton.addEventListener("click", createNewChat);
    clearHistoryButton.addEventListener("click", clearAllHistory);
    toggleThemeButton.addEventListener("click", toggleTheme);
    exportChatButton.addEventListener("click", exportCurrentChat);
    regenerateResponseButton.addEventListener("click", regenerateLastResponse);
    stopResponseButton.addEventListener("click", () => {
      stopGeneration = true;
      clearTimeout(letterTimeout);
      stopResponseButton.style.display = "none";
      regenerateResponseButton.style.display = "inline-block";
    });

    fileUploadButton.addEventListener("click", () => fileUploadInput.click());
    fileUploadInput.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) {
        pendingFile = file;
        displayPendingFilePreview(file);
      }
    });

    suggestionChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        userInput.value = chip.dataset.prompt || chip.textContent;
        handleSendMessage();
      });
    });

    // History
    updateChatHistorySidebar();
    if (Object.keys(chatHistory).length === 0) {
      createNewChat();
    } else {
      const mostRecentChatId = Object.keys(chatHistory).sort((a, b) => {
        return chatHistory[b].timestamp - chatHistory[a].timestamp;
      })[0];
      loadChat(mostRecentChatId);
    }

    // Close any open 3-dot menus
    window.addEventListener("click", () => {
      document.querySelectorAll(".chat-options-menu").forEach((menu) => {
        menu.style.display = "none";
      });
    });
  }

  // ---------- File preview helpers ----------
  function displayPendingFilePreview(file) {
    const previewContainer = document.getElementById("pending-file-preview");
    const reader = new FileReader();
    reader.onload = function (e) {
      let previewHTML = "";
      if (file.type.startsWith("image/")) {
        previewHTML = `<img src="${e.target.result}" alt="${file.name}" style="max-width: 100px; max-height: 100px;"/>`;
      } else if (file.type.startsWith("text/") || file.type === "application/json") {
        let content = e.target.result;
        if (content.length > 200) content = content.substring(0, 200) + "...";
        previewHTML = `<pre style="white-space: pre-wrap; font-size: 12px;">${escapeHtml(content)}</pre>`;
      } else {
        previewHTML = `<div style="font-size: 12px;">${file.name}</div>`;
      }
      previewContainer.innerHTML = previewHTML;
      previewContainer.style.display = "block";
    };
    if (file.type.startsWith("image/")) reader.readAsDataURL(file);
    else reader.readAsText(file);
  }

  // Bu fonksiyon sadece UI'ya dosya önizlemesi ekler ve HISTORY'ye kaydeder.
  // Not: pendingFile'ı burada SIFIRLAMAYACAĞIZ; detect için lazım.
  function processPendingFile() {
    return new Promise((resolve) => {
      const file = pendingFile;
      if (!file) { resolve(); return; }
      const reader = new FileReader();
      reader.onload = function (e) {
        let previewHTML = "";
        if (file.type.startsWith("image/")) {
          previewHTML = `<img src="${e.target.result}" alt="${file.name}" style="max-width:100%;"/>`;
        } else if (file.type.startsWith("text/") || file.type === "application/json") {
          previewHTML = `<pre style="white-space: pre-wrap;">${escapeHtml(e.target.result)}</pre>`;
        } else {
          previewHTML = `<div>Uploaded file: ${file.name}</div>`;
        }
        addFileMessageToUI("user", previewHTML);
        if (!chatHistory[currentChatId]) createNewChat();
        chatHistory[currentChatId].messages.push({
          role: "user",
          content: previewHTML,
          file: { name: file.name, type: file.type }
        });
        const previewContainer = document.getElementById("pending-file-preview");
        previewContainer.style.display = "none";
        previewContainer.innerHTML = "";
        resolve();
      };
      if (file.type.startsWith("image/")) reader.readAsDataURL(file);
      else reader.readAsText(file);
    });
  }

  function escapeHtml(text) {
    var map = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"};
    return text.replace(/[&<>"']/g, m => map[m]);
  }

  function addFileMessageToUI(sender, htmlContent) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.innerHTML = htmlContent;
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function handleSendMessage() {
    if (isTyping) {
      alert("Please wait until the current response is completed.");
      return;
    }

    const message = userInput.value.trim();
    const hasFile = !!pendingFile;
    if (!message && !hasFile) return;

    // input'u temizle
    userInput.value = "";
    userInput.style.height = "auto";

    // UI: user text
    if (message) {
      addMessageToUI("user", message);
      if (!chatHistory[currentChatId]) createNewChat();
      chatHistory[currentChatId].messages.push({ role: "user", content: message });
    }

    // Eğer dosya seçiliyse: önce önizlemeyi UI'a bas
    let fileRef = null;
    if (hasFile) {
      fileRef = pendingFile;        // detect'te kullanacağız
      processPendingFile();         // sadece UI/hafıza (await etmeye gerek yok)
   }

    saveChatHistory();

    //   Asıl işleyen pipeline (intent + komut + RAG) burası
    runMessagePipeline(message, fileRef);
  }

  function buildRunMetadata() {
    return {
      use_internet: webSearchEnabled,
      web_only: false,
    };
  }

  function appendAssistantResponse(content) {
    const aiText = (content || "").trim() || "(Empty Answer)";
    if (typeof addFormattedMessageToUI === "function") {
      addFormattedMessageToUI("ai", aiText);
    } else {
      addMessageToUI("ai", aiText);
    }
    if (!chatHistory[currentChatId]) createNewChat();
    chatHistory[currentChatId].messages.push({ role: "assistant", content: aiText });
    saveChatHistory();
  }

  function logRunDebug(result) {
    if (!result) return;
    console.info("pipeline result", {
      status: result.status,
      route: result.route?.route,
      intent: result.intent?.label,
      client_action: result.client_action?.action,
      duration_ms: result.duration_ms,
    });
    if (Array.isArray(result.warnings) && result.warnings.length) {
      console.warn("pipeline warnings", result.warnings);
    }
    updateTelemetryBar(result);
  }

  function updateTelemetryBar(result) {
    const bar = document.getElementById('telemetry-bar');
    if (!bar || !result) return;
    bar.innerHTML = '';
    const badges = [];

    // Status
    if (result.status) {
      const cls = result.status === 'completed' ? 'status-completed' :
                  result.status === 'degraded' ? 'status-degraded' : 'status-failed';
      const icon = result.status === 'completed' ? 'fa-check-circle' :
                   result.status === 'degraded' ? 'fa-exclamation-triangle' : 'fa-times-circle';
      badges.push(`<span class="telemetry-badge ${cls}"><i class="fas ${icon}"></i>${result.status}</span>`);
    }

    // Route
    if (result.route?.route) {
      const r = result.route.route;
      const cls = r === 'chat' ? 'route-chat' : r === 'rag' ? 'route-rag' :
                  r === 'detect' ? 'route-detect' : 'route-camera';
      const icon = r === 'chat' ? 'fa-comment' : r === 'rag' ? 'fa-database' :
                   r === 'detect' ? 'fa-crosshairs' : 'fa-video';
      badges.push(`<span class="telemetry-badge ${cls}"><i class="fas ${icon}"></i>${r}</span>`);
    }

    // Intent confidence
    if (result.intent?.label && result.intent?.confidence != null) {
      const pct = Math.round(result.intent.confidence * 100);
      badges.push(`<span class="telemetry-badge"><i class="fas fa-brain"></i>${result.intent.label} ${pct}%</span>`);
    }

    // Retrieval
    if (result.retrieval?.used_context) {
      const mode = result.retrieval.retrieval_mode || 'local';
      badges.push(`<span class="telemetry-badge route-rag"><i class="fas fa-search"></i>retrieval: ${mode}</span>`);
    }

    // Generation model
    if (result.generation?.model_name) {
      const rt = result.generation.runtime || '';
      badges.push(`<span class="telemetry-badge"><i class="fas fa-microchip"></i>${result.generation.model_name}${rt ? ' \u00b7 ' + rt : ''}</span>`);
    }

    // Duration
    if (result.duration_ms != null) {
      badges.push(`<span class="telemetry-badge"><i class="fas fa-clock"></i>${result.duration_ms}ms</span>`);
    }

    if (badges.length > 0) {
      bar.innerHTML = badges.join('');
      bar.classList.add('visible');
    } else {
      bar.classList.remove('visible');
    }
  }

  function formatDetectionResponse(det) {
    const summary = det?.summary || "no objects";
    let text = `${det?.narration ? det.narration + "\n\n" : ""}Object Summary: ${summary}`;
    if (det?.image_url) text += `\n\n(Image: ${det.image_url})`;
    if (det?.error) console.warn("detection warning", det.error);
    return text;
  }

  function formatPhotoResponse(res) {
    if (res?.image_url) {
      return `Photo saved and available at ${res.image_url}. I've also sent it to your phone.`;
    }
    return "Photo saved. I've also sent it to your phone.";
  }

  function shouldDetectAfterCapture(result) {
    const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
    const reasonText = [
      result?.route?.route,
      result?.route?.fallback_reason,
      result?.client_action?.reason,
      result?.detection?.error,
    ].filter(Boolean).join(" ");

    return (
      result?.route?.route === "detect" ||
      warnings.includes("missing_image_for_detection") ||
      reasonText.includes("missing_image_for_detection")
    );
  }

  function waitForCameraFrame(timeoutMs = 1500) {
    return new Promise((resolve) => {
      if (!videoEl || (videoEl.readyState >= 2 && videoEl.videoWidth > 0)) {
        resolve();
        return;
      }

      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        videoEl.removeEventListener("loadedmetadata", finish);
        videoEl.removeEventListener("canplay", finish);
        resolve();
      };

      videoEl.addEventListener("loadedmetadata", finish, { once: true });
      videoEl.addEventListener("canplay", finish, { once: true });
      setTimeout(finish, timeoutMs);
    });
  }

  async function captureCameraFile(filename) {
    if (!mediaStream) {
      await openCamera();
    }
    await waitForCameraFrame();
    const blob = await takePhotoBlob();
    if (!blob) throw new Error("Could not capture a photo.");
    return new File([blob], filename, { type: blob.type || "image/jpeg" });
  }

  async function executeClientAction(clientAction, result, context = {}) {
    const action = typeof clientAction === "string" ? clientAction : clientAction?.action;
    if (!action || action === "none") return "";

    if (action === "open_camera") {
      const msg = await openCamera();
      return result?.final_answer ? "" : (msg || "The camera is already open.");
    }

    if (action === "close_camera") {
      const msg = closeCamera();
      return result?.final_answer ? "" : msg;
    }

    if (action === "capture_photo") {
      const shouldDetect = shouldDetectAfterCapture(result);
      if (shouldDetect) {
        const file = context.fileRef || await captureCameraFile("frame.jpg");
        const det = await apiDetectFromFile(file, 1);
        return formatDetectionResponse(det);
      }

      const file = await captureCameraFile("snapshot.jpg");
      const res = await apiPhotoFromFile(file);
      return formatPhotoResponse(res);
    }

    console.warn("Unsupported client action", action);
    return "";
  }

  async function handleRunResult(result, context = {}) {
    logRunDebug(result);
    const messages = [];

    if (result?.final_answer) {
      messages.push(result.final_answer);
    }

    if (result?.client_action) {
      const actionMessage = await executeClientAction(result.client_action, result, context);
      if (actionMessage) messages.push(actionMessage);
    }

    if (!messages.length) {
      if (Array.isArray(result?.errors) && result.errors.length) {
        messages.push("Sorry, the backend could not complete the request.");
      } else if (Array.isArray(result?.warnings) && result.warnings.length) {
        messages.push("The request completed with a warning, but no answer was returned.");
      } else {
        messages.push("(Empty Answer)");
      }
    }

    messages.forEach(appendAssistantResponse);
  }

  async function runLegacyMessagePipeline(message, fileRef) {
    let aiText = "";

    if (fileRef) {
      const det = await apiDetectFromFile(fileRef, 1);
      return formatDetectionResponse(det);
    }

    const intentInput = (message || "")
      .trim()
      .replace(/[.!?]+$/g, "");

    const { intent, score, threshold, narration } = await apiIntent(intentInput);
    const THR = typeof threshold === "number" ? threshold : 0.7;
    const words = (message || "").split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const looksLikeCommand = wordCount <= 7 && !message.includes("?");

    if (
      ["open_camera", "close_camera", "take_photo", "object_detect"].includes(intent) &&
      score >= THR &&
      looksLikeCommand
    ) {
      if (intent === "open_camera") {
        const msg = await openCamera();
        aiText = narration || msg || "The camera is already open.";
      } else if (intent === "close_camera") {
        const msg = closeCamera();
        aiText = narration || msg;
      } else if (intent === "take_photo") {
        const file = await captureCameraFile("snapshot.jpg");
        const res = await apiPhotoFromFile(file);
        aiText = formatPhotoResponse(res);
      } else if (intent === "object_detect") {
        if (mediaStream) {
          const file = await captureCameraFile("frame.jpg");
          const det = await apiDetectFromFile(file, 1);
          aiText = formatDetectionResponse(det);
        } else {
          aiText = "For object detection, please upload an image or say 'open camera', 'Object detect'.";
        }
      }
    } else if (intent === "chat" && score >= THR) {
      if (webSearchEnabled) {
        const res = await apiRag(message, true, true);
        aiText = res.answer || "(Empty Answer)";
        if (res.used_context && Array.isArray(res.sources) && res.sources.length) {
          aiText += "\n\nResources:\n- " + res.sources.join("\n- ");
        }
      } else {
        const res = await apiChat(message);
        aiText = res.answer || "(Empty Answer)";
      }
    } else {
      const res = await apiRag(message, webSearchEnabled, false);
      aiText = res.answer || "(Empty Answer)";
      if (res.used_context && Array.isArray(res.sources) && res.sources.length) {
        aiText += "\n\nResources:\n- " + res.sources.join("\n- ");
      }
    }

    return aiText;
  }

  async function runMessagePipeline(message, fileRef) {
    try {
      showTypingIndicator();
      isTyping = true;

      if (fileRef && !(message || "").trim()) {
        const det = await apiDetectFromFile(fileRef, 1);
        appendAssistantResponse(formatDetectionResponse(det));
        return;
      }

      let result = null;
      try {
        result = await apiRun(message, buildRunMetadata());
      } catch (runError) {
        console.warn("/api/run failed; using legacy frontend fallback.", runError);
        const aiText = await runLegacyMessagePipeline(message, fileRef);
        appendAssistantResponse(aiText);
        return;
      }

      await handleRunResult(result, { fileRef, message });
    } catch (err) {
      console.error(err);
      appendAssistantResponse("Sorry, an error occurred. Please try again.");
    } finally {
      removeTypingIndicator();
      isTyping = false;
      pendingFile = null;
    }
  }


  // ---------- UI helpers ----------
  function addMessageToUIWithTypingEffect(sender, content) {
    removeTypingIndicator();
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;

    if (sender === "user") {
      const messageContent = document.createElement("div");
      messageContent.className = "message-content";
      messageContent.textContent = content;
      messageDiv.appendChild(messageContent);
      messagesContainer.appendChild(messageDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      return;
    }

    // AI avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-anchor"></i>';
    messageDiv.appendChild(avatar);

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);

    const processedContent = processMarkdownContent(content);
    if (sender === "ai") {
      stopGeneration = false;
      stopResponseButton.style.display = "inline-block";
    }
    startTypingEffect(messageContent, processedContent, 0);
  }

  function addMessageToUI(sender, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    if (sender === 'ai') {
      const avatar = document.createElement('div');
      avatar.className = 'message-avatar';
      avatar.innerHTML = '<i class="fas fa-anchor"></i>';
      messageDiv.appendChild(avatar);
    }
    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.textContent = content;
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function processMarkdownContent(content) {
    const segments = [];
    let currentPos = 0;
    const codeBlockRegex = /```([\w]*)\n([\s\S]*?)\n```/g;
    let match;
    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > currentPos) {
        segments.push({ type: "text", content: content.substring(currentPos, match.index) });
      }
      segments.push({ type: "code", language: match[1] || "plaintext", content: match[2] });
      currentPos = match.index + match[0].length;
    }
    if (currentPos < content.length) {
      segments.push({ type: "text", content: content.substring(currentPos) });
    }
    return segments;
  }

  function startTypingEffect(messageContent, segments, segmentIndex) {
    if (segmentIndex >= segments.length) {
      isTyping = false;
      stopResponseButton.style.display = "none";
      return;
    }
    const segment = segments[segmentIndex];

    if (segment.type === "code") {
      const preElement = document.createElement("pre");
      const codeElement = document.createElement("code");
      if (segment.language) codeElement.className = `language-${segment.language}`;
      codeElement.classList.add("hljs");

      const copyButtonContainer = document.createElement("div");
      copyButtonContainer.className = "code-copy-container";
      const copyButton = document.createElement("button");
      copyButton.className = "code-copy-button";
      copyButton.innerHTML = '<i class="fas fa-copy"></i>';
      copyButton.title = "Copy code";
      copyButton.addEventListener("click", () => {
        navigator.clipboard.writeText(segment.content).then(() => {
          copyButton.innerHTML = '<i class="fas fa-check"></i>';
          copyButton.classList.add("copied");
          setTimeout(() => {
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.classList.remove("copied");
          }, 2000);
        });
      });

      copyButtonContainer.appendChild(copyButton);
      preElement.appendChild(copyButtonContainer);
      preElement.appendChild(codeElement);
      messageContent.appendChild(preElement);

      typeCodeContent(codeElement, segment.content, 0, () => {
        hljs.highlightElement(codeElement);
        startTypingEffect(messageContent, segments, segmentIndex + 1);
      });
    } else {
      const textDiv = document.createElement("div");
      messageContent.appendChild(textDiv);
      typeTextContent(textDiv, segment.content, 0, () => {
        startTypingEffect(messageContent, segments, segmentIndex + 1);
      });
    }
  }

  function typeCodeContent(element, content, index, callback) {
    if (stopGeneration) { callback(); return; }
    if (index < content.length) {
      element.textContent += content[index];
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      letterTimeout = setTimeout(() => {
        typeCodeContent(element, content, index + 1, callback);
      }, typingSpeed);
    } else {
      callback();
    }
  }

  function typeTextContent(element, content, index, callback) {
    if (stopGeneration) { callback(); return; }
    if (index < content.length) {
      let currentText = content.substring(0, index + 1);
      element.innerHTML = marked.parse(currentText);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      letterTimeout = setTimeout(() => {
        typeTextContent(element, content, index + 1, callback);
      }, typingSpeed);
    } else {
      callback();
    }
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "typing-indicator";
    typingDiv.id = "typing-indicator";
    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'typing-avatar';
    avatar.innerHTML = '<i class="fas fa-anchor"></i>';
    typingDiv.appendChild(avatar);
    // Dots wrapper
    const dotsWrapper = document.createElement('div');
    dotsWrapper.className = 'typing-dots';
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement("div");
      dot.className = "typing-dot";
      dotsWrapper.appendChild(dot);
    }
    typingDiv.appendChild(dotsWrapper);
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function removeTypingIndicator() {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) typingIndicator.remove();
  }

  function createNewChat() {
    const chatId = "chat_" + Date.now();
    chatHistory[chatId] = { id: chatId, title: "New Conversation", timestamp: Date.now(), messages: [] };
    currentChatId = chatId;
    currentChatTitle.textContent = "New Conversation";
    messagesContainer.innerHTML = `
      <div class="intro-message">
        <div class="intro-icon"><i class="fas fa-ship"></i></div>
        <h1>Welcome to <span class="brand-accent">PathFinderShip</span></h1>
        <p>Your local AI maritime assistant \u2014 ask questions, detect objects, upload documents, or control the camera.</p>
        <div class="suggestion-chips">
          <button class="suggestion-chip" data-prompt="Can you introduce yourself?"><i class="fas fa-comment" style="margin-right:6px;opacity:0.5"></i>Can you introduce yourself?</button>
          <button class="suggestion-chip" data-prompt="Open the camera"><i class="fas fa-camera" style="margin-right:6px;opacity:0.5"></i>Open the camera</button>
          <button class="suggestion-chip" data-prompt="Detect objects"><i class="fas fa-crosshairs" style="margin-right:6px;opacity:0.5"></i>Detect objects</button>
          <button class="suggestion-chip" data-prompt="What can you do?"><i class="fas fa-search" style="margin-right:6px;opacity:0.5"></i>What can you do?</button>
        </div>
      </div>`;
    document.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        userInput.value = chip.dataset.prompt || chip.textContent;
        handleSendMessage();
      });
    });
    saveChatHistory();
    updateChatHistorySidebar();
    pendingFile = null;
    // Clear telemetry
    const tbar = document.getElementById('telemetry-bar');
    if (tbar) { tbar.innerHTML = ''; tbar.classList.remove('visible'); }
    const previewContainer = document.getElementById("pending-file-preview");
    if (previewContainer) { previewContainer.innerHTML = ""; previewContainer.style.display = "none"; }
  }

  function loadChat(chatId) {
    if (!chatHistory[chatId]) return;
    currentChatId = chatId;
    currentChatTitle.textContent = chatHistory[chatId].title;
    messagesContainer.innerHTML = "";
    chatHistory[chatId].messages.forEach((message) => {
      if (message.role === "user") {
        if (message.file) addFileMessageToUI("user", message.content);
        else addMessageToUI("user", message.content);
      } else {
        addFormattedMessageToUI("ai", message.content);
      }
    });
    updateActiveChatInSidebar();
  }

  function addFormattedMessageToUI(sender, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    if (sender === 'ai') {
      const avatar = document.createElement('div');
      avatar.className = 'message-avatar';
      avatar.innerHTML = '<i class="fas fa-anchor"></i>';
      messageDiv.appendChild(avatar);
    }
    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    const processedContent = processMarkdownContent(content);
    processedContent.forEach((segment) => {
      if (segment.type === "code") {
        const preElement = document.createElement("pre");
        const codeElement = document.createElement("code");
        if (segment.language) codeElement.className = `language-${segment.language}`;
        codeElement.classList.add("hljs");

        const copyButtonContainer = document.createElement("div");
        copyButtonContainer.className = "code-copy-container";
        const copyButton = document.createElement("button");
        copyButton.className = "code-copy-button";
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = "Copy code";
        copyButton.addEventListener("click", () => {
          navigator.clipboard.writeText(segment.content).then(() => {
            copyButton.innerHTML = '<i class="fas fa-check"></i>';
            copyButton.classList.add("copied");
            setTimeout(() => {
              copyButton.innerHTML = '<i class="fas fa-copy"></i>';
              copyButton.classList.remove("copied");
            }, 2000);
          });
        });
        copyButtonContainer.appendChild(copyButton);
        preElement.appendChild(copyButtonContainer);
        codeElement.textContent = segment.content;
        preElement.appendChild(codeElement);
        messageContent.appendChild(preElement);
        hljs.highlightElement(codeElement);
      } else {
        const textDiv = document.createElement("div");
        textDiv.innerHTML = marked.parse(segment.content);
        while (textDiv.firstChild) messageContent.appendChild(textDiv.firstChild);
      }
    });
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function saveChatHistory() {
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    updateChatHistorySidebar();
  }

  function updateChatHistorySidebar() {
    chatHistoryContainer.innerHTML = "";
    const sortedChatIds = Object.keys(chatHistory).sort((a, b) => chatHistory[b].timestamp - chatHistory[a].timestamp);
    sortedChatIds.forEach((chatId) => {
      const chat = chatHistory[chatId];
      const chatItem = document.createElement("div");
      chatItem.className = `chat-history-item ${chatId === currentChatId ? "active" : ""}`;
      chatItem.dataset.chatId = chatId;

      const icon = document.createElement("i"); icon.className = "fas fa-comment";
      const titleSpan = document.createElement("span"); titleSpan.textContent = chat.title;
      chatItem.appendChild(icon); chatItem.appendChild(titleSpan);

      chatItem.addEventListener("click", () => loadChat(chatId));

      const optionsButton = document.createElement("button");
      optionsButton.className = "chat-options-button";
      optionsButton.innerHTML = '<i class="fas fa-ellipsis-v"></i>';
      optionsButton.addEventListener("click", (e) => {
        e.stopPropagation();
        optionsMenu.style.display = optionsMenu.style.display === "none" ? "block" : "none";
      });

      const optionsMenu = document.createElement("div");
      optionsMenu.className = "chat-options-menu";
      optionsMenu.style.display = "none";
      optionsMenu.innerHTML = `
        <div class="chat-options-item delete-chat">Delete</div>
        <div class="chat-options-item rename-chat">Rename</div>
      `;

      optionsMenu.querySelector(".delete-chat").addEventListener("click", (e) => {
        e.stopPropagation();
        if (confirm("Are you sure you want to delete this chat?")) {
          delete chatHistory[chatId];
          if (currentChatId === chatId) createNewChat();
          saveChatHistory();
          updateChatHistorySidebar();
        }
      });

      optionsMenu.querySelector(".rename-chat").addEventListener("click", (e) => {
        e.stopPropagation();
        const newName = prompt("Enter new name for this chat:", chat.title);
        if (newName) {
          chat.title = newName;
          saveChatHistory();
          updateChatHistorySidebar();
          if (currentChatId === chatId) currentChatTitle.textContent = newName;
        }
      });

      chatItem.appendChild(optionsButton);
      chatItem.appendChild(optionsMenu);
      chatHistoryContainer.appendChild(chatItem);
    });
  }

  function updateActiveChatInSidebar() {
    document.querySelectorAll(".chat-history-item").forEach((item) => {
      item.classList.remove("active");
      if (item.dataset.chatId === currentChatId) item.classList.add("active");
    });
  }

  function clearAllHistory() {
    if (confirm("Are you sure you want to clear all chat history? This cannot be undone.")) {
      chatHistory = {};
      localStorage.removeItem("chatHistory");
      createNewChat();
    }
  }

  function toggleTheme() {
    if (currentTheme === "light") {
      document.body.classList.add("dark-mode");
      currentTheme = "dark";
      toggleThemeButton.innerHTML = '<i class="fas fa-sun"></i><span>Light Mode</span>';
    } else {
      document.body.classList.remove("dark-mode");
      currentTheme = "light";
      toggleThemeButton.innerHTML = '<i class="fas fa-moon"></i><span>Dark Mode</span>';
    }
    localStorage.setItem("theme", currentTheme);
  }

  function exportCurrentChat() {
    if (!chatHistory[currentChatId]) return;
    const chat = chatHistory[currentChatId];
    let exportText = `# ${chat.title}\n\n`;
    chat.messages.forEach((message) => {
      const role = message.role === "user" ? "You" : "PathFinderShip";
      exportText += `## ${role}:\n${message.content}\n\n`;
    });
    const blob = new Blob([exportText], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${chat.title.replace(/[^\w\s]/gi, "")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Son mesajı backend ile yeniden üret (OpenRouter yok)
  async function regenerateLastResponse() {
    const chat = chatHistory[currentChatId];
    if (!chat || chat.messages.length === 0) return;

    // Son AI cevabını sil ve son kullanıcı mesajını bul
    const last = chat.messages[chat.messages.length - 1];
    if (last.role !== "assistant") return;

    // UI'dan da sil
    const aiMsgs = document.querySelectorAll(".message.ai");
    if (aiMsgs.length > 0) aiMsgs[aiMsgs.length - 1].remove();
    chat.messages.pop();
    saveChatHistory();

    // Son user mesajını bul
    let i = chat.messages.length - 1;
    let lastUser = null;
    for (; i >= 0; i--) {
      if (chat.messages[i].role === "user" && !chat.messages[i].file) {
        lastUser = chat.messages[i].content;
        break;
      }
    }
    if (!lastUser) { addFormattedMessageToUI("ai", "Yeniden \u00fcretmek i\u00e7in metin mesaj bulunamad\u0131."); return; }

    // Yeniden çağır: ana mesaj akışı gibi /api/run kullanır.
    await runMessagePipeline(lastUser, null);
  }

  function autoResizeTextarea() {
    userInput.style.height = "auto";
    userInput.style.height = userInput.scrollHeight + "px";
  }
});
