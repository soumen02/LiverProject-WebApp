window.validateForm = function (e, formElement) {
    console.log('validateForm');
    var fileInput = formElement.querySelector('input[type="file"]');
    var errorElement = formElement.querySelector('#error');
    if (fileInput.files.length === 0) {
        e.preventDefault();
        errorElement.style.display = 'block';
    } else {
        errorElement.style.display = 'none';
    }
}

