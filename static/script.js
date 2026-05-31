function createMessage(text, className) {
    const msg = document.createElement("div");
    msg.className = "message " + className;
    msg.innerText = text;
    document.getElementById("chatBox").appendChild(msg);
    document.getElementById("chatBox").scrollTop =
        document.getElementById("chatBox").scrollHeight;
    return msg;
}

async function sendMessage() {

    const input = document.getElementById("userInput");
    const text = input.value.trim();

    if (!text) return;

    input.value = "";

    // Add user bubble
    createMessage(text, "user");

    // Create bot bubble
    const botMsg = createMessage("", "bot");

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: text })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        botMsg.innerText += chunk;

        document.getElementById("chatBox").scrollTop =
            document.getElementById("chatBox").scrollHeight;
    }
}