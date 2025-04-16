const chatMessagesDiv = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const fileInput = document.getElementById("fileInput");
const fileBtn = document.getElementById("fileBtn");
const sendBtn = document.getElementById("sendBtn");
const filePreviewArea = document.getElementById("filePreviewArea");

let selectedFiles = [];

fileBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  for (const file of Array.from(fileInput.files)) {
    const duplicate = selectedFiles.find(f => f.name === file.name && f.size === file.size);
    if (!duplicate) selectedFiles.push(file);
  }
  renderFilePreview();
  fileInput.value = ""; // Clear to allow re-upload of same file
});

function renderFilePreview() {
  filePreviewArea.innerHTML = "";
  selectedFiles.forEach((file, index) => {
    const tag = document.createElement("div");
    tag.className = "file-tag";

    const icon = document.createElement("i");
    icon.className = "bi bi-file-earmark";
    if (file.name.endsWith(".pdf")) icon.classList.add("text-danger");
    else if (file.name.match(/\.(csv|xlsx|xls)$/i)) icon.classList.add("text-success");
    else if (file.name.match(/\.(doc|docx)$/i)) icon.classList.add("text-primary");

    const name = document.createElement("span");
    name.textContent = file.name;

    const close = document.createElement("button");
    close.innerHTML = "&times;";
    close.className = "file-remove-btn";
    close.onclick = () => {
      selectedFiles.splice(index, 1);
      renderFilePreview();
    };

    tag.appendChild(icon);
    tag.appendChild(name);
    tag.appendChild(close);
    filePreviewArea.appendChild(tag);
  });
}

sendBtn.addEventListener("click", async () => {
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage({ text, sender: "user", files: selectedFiles });

  const formData = new FormData();
  formData.append("user_text", text);
  selectedFiles.forEach(file => formData.append("uploaded_file", file));

  userInput.value = "";
  selectedFiles = [];
  renderFilePreview();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      body: formData,
      credentials: "include"
    });

    const contentType = res.headers.get("content-type");
    if (!res.ok || !contentType.includes("application/json")) {
      throw new Error("Unexpected server response (non-JSON).");
    }

    const data = await res.json();
    appendMessage({ text: data.reply, sender: "ai" });

  } catch (err) {
    console.error(err);
    appendMessage({ text: `❗ Server error: ${err.message}`, sender: "ai" });
  }
});

function appendMessage({ text, sender, files = [] }) {
  const msgCard = document.createElement("div");
  msgCard.classList.add("msg-card", sender === "user" ? "msg-user" : "msg-ai");
  msgCard.style.maxWidth = "100%";
  msgCard.style.overflowWrap = "break-word";
  msgCard.style.whiteSpace = "pre-wrap";

  const content = document.createElement("div");
  content.innerHTML = formatMessageText(text);
  msgCard.appendChild(content);

  if (sender === "user") {
    files.forEach(file => {
      const fileDiv = document.createElement("div");
      fileDiv.classList.add("file-preview-box");
      fileDiv.innerHTML = `<i class="bi bi-paperclip"></i> ${file.name}`;
      msgCard.appendChild(fileDiv);
    });
  }

  chatMessagesDiv.appendChild(msgCard);
  chatMessagesDiv.scrollTo({
    top: chatMessagesDiv.scrollHeight,
    behavior: "smooth"
  });
}

function formatMessageText(text) {
  try {
    const codeBlockRegex = /```([a-z]*)\n([\s\S]*?)```/g;

    const formatted = text.replace(codeBlockRegex, (match, lang, code) => {
      return `<pre class="code-block"><code class="language-${lang || 'plaintext'}">${escapeHTML(code.trim())}</code></pre>`;
    });

    return formatted.replace(/\n/g, "<br>");
  } catch (e) {
    console.error("Format error:", e);
    return escapeHTML(text).replace(/\n/g, "<br>");
  }
}

function escapeHTML(str) {
  return str.replace(/[&<>"']/g, tag => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[tag]
  ));
}

function changeModel(value) {
  let model = "gpt4o";
  let groq_model = "";

  if (value.startsWith("groq")) {
    model = "groq";

    if (value === "groq_llama4") {
      groq_model = "meta-llama/llama-4-maverick-17b-128e-instruct";
    } else if (value === "groq_deepseek") {
      groq_model = "deepseek-r1-distill-qwen-32b";
    } else if (value === "groq_mistral24b") {
      groq_model = "mistral-saba-24b";
    } else {
      groq_model = "llama3-70b-8192";
    }
  }

  fetch("/set_model", {
    method: "POST",
    body: new URLSearchParams({ model, groq_model }),
    credentials: "include"
  }).then(res => res.json());
}
