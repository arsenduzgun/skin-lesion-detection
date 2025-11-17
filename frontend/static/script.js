const form = document.getElementById('upload-form');
const resultText = document.getElementById('result-text');
const processingLoader = document.getElementById('processing-loader');


form.addEventListener('submit', function (event) {
  event.preventDefault();

  const formData = new FormData(form);
  // show loader, clear any previous text
  processingLoader.style.display = 'flex';
  resultText.textContent = '';

  fetch('/predict', {
    method: 'POST',
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {

      processingLoader.style.display = 'none';

      if (data.prediction) {
        resultText.innerHTML = `<strong>Prediction:&nbsp;</strong>${data.prediction}`;
      } else if (data.error) {
        resultText.innerHTML = `<strong>Error:&nbsp;</strong>${data.error}`;
      } else {
        resultText.innerHTML = `<strong>Error:&nbsp;</strong>No prediction received.`;
      }
    })
    .catch(() => {
      processingLoader.style.display = 'none';
      resultText.innerHTML = `<strong>Error:&nbsp;</strong>Failed to fetch prediction.`;
    });
});
