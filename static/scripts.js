document.getElementById("send-btn").addEventListener("click", function() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    // Display the user's message in the chat box
    displayMessage(userInput, "user-message");

    // Send the user's input to the FastAPI server
    fetch("/query/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Display the bot's response in the chat box, excluding the YouTube link
        const cleanedResponse = data.chatgpt_response.replace(/https?:\/\/[^\s]+/, '');
        displayMessage(cleanedResponse, "bot-message");

        // Embed the YouTube video
        embedYouTubeVideo(data.youtube_link);
    });

    // Clear the input field
    document.getElementById("user-input").value = "";
});

function displayMessage(message, className) {
    const chatBox = document.getElementById("chat-box");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${className}`;
    messageDiv.innerHTML = message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function embedYouTubeVideo(youtubeLink) {
    const videoContainer = document.getElementById("video-container");
    const videoId = youtubeLink.split("v=")[1];
    if (videoId) {
        const embedUrl = `https://www.youtube.com/embed/${videoId}`;
        videoContainer.innerHTML = `<iframe src="${embedUrl}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`;
    }
}
