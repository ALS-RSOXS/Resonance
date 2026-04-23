window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

const typesetMath = () => {
  if (!window.MathJax || !window.MathJax.typesetPromise) {
    return;
  }
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
};

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(typesetMath);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", typesetMath);
} else {
  typesetMath();
}
