const form = document.getElementById('upload-form');
const result = document.querySelector('.result');

form.addEventListener('submit', function (event) {
  event.preventDefault();

  const formData = new FormData(form);

  fetch('/predict', {
    method: 'POST',
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.prediction) {
        result.innerHTML = `<strong>Prediction:&nbsp;</strong> ${data.prediction}`;
      } else {
        result.innerHTML = `<strong>Error:&nbsp;</strong> No prediction received.`;
      }
    })
    .catch(() => {
      result.innerHTML = `<strong>Error:&nbsp;</strong> Failed to fetch prediction.`;
    });
});
