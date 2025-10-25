async function analyze() {
    const review = document.getElementById("review").value;

    if (!review) {
        alert("Please write a review first");
        return;
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review: review }),
    });

    const result = await response.json();
    console.log(result); // should show {sentiment: "Positive"} or {sentiment: "Negative"}

    // ✅ use backticks here
    alert(`The sentiment is: ${result.sentiment}`);
}
