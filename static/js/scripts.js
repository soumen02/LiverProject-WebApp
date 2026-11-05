window.validateForm = function (e, formElement) {
    console.log('validateForm');
    var fileInput = formElement.querySelector('input[type="file"]');
    var errorElement = formElement.querySelector('#error');
    if (fileInput.files.length === 0) {
        e.preventDefault();
        errorElement.style.display = 'block';
    } else {
        errorElement.style.display = 'none';
        // segmenting takes a few minutes on a CPU, so show that something is happening
        var button = formElement.querySelector('button[type="submit"]');
        button.disabled = true;
        button.textContent = 'Segmenting...';
    }
}

