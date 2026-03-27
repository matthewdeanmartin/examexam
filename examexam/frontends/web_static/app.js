document.addEventListener("DOMContentLoaded", () => {
  const autofocusTarget = document.querySelector("[data-autofocus]");
  if (autofocusTarget instanceof HTMLElement) {
    autofocusTarget.focus();
  }
});
