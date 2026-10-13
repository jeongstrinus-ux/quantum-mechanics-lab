// Dash commits direct slider entry on blur; Enter should commit it too.
document.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && event.target instanceof HTMLInputElement &&
        event.target.matches('.dash-slider-container .dash-range-slider-input')) {
        event.preventDefault();
        event.target.blur();
    }
});
