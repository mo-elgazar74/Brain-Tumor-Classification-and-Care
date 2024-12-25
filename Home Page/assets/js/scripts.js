document.addEventListener("DOMContentLoaded", () => {
    const faqItems = document.querySelectorAll(".faq-item");

    faqItems.forEach(item => {
        const questionButton = item.querySelector(".faq-question");

        questionButton.addEventListener("click", () => {
            const answer = item.querySelector(".faq-answer");

            item.classList.toggle("active");

            if (answer.style.display === "block") {
                answer.style.display = "none";
                questionButton.classList.remove("active");
            } else {
                answer.style.display = "block";
                questionButton.classList.add("active");
            }
        });
    });
});
