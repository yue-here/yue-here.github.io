# Taking the diffraction pattern of a city


<div class="slider-container unique-slider-container1">
  <button id="autoplayButton1" class="autoplayButton">Pause</button>
  <span id="sliderValue1" class="sliderValue">CFG scale 1</span>
  <input type="range" min="0" max="99" step="1" value="1" id="parameterSlider1" class="parameterSlider">
</div>
<img id="outputImage1" class="outputImage" src="/fire_slider/fire-CFG0_grid.png" alt="Model Output">

<div class="slider-container unique-slider-container2">
  <button id="autoplayButton2" class="autoplayButton">Pause</button>
  <span id="sliderValue2" class="sliderValue">CFG scale 1</span>
  <input type="range" min="0" max="99" step="1" value="1" id="parameterSlider2" class="parameterSlider">
</div>
<img id="outputImage2" class="outputImage" src="/bird_slider/bird-CFG0_grid.png" alt="Model Output">

<div class="slider-container unique-slider-container3">
  <button id="autoplayButton3" class="autoplayButton">Pause</button>
  <span id="sliderValue3" class="sliderValue">CFG scale 1</span>
  <input type="range" min="0" max="99" step="1" value="1" id="parameterSlider3" class="parameterSlider">
</div>
<img id="outputImage3" class="outputImage" src="/hair_slider/hair-CFG0_grid.png" alt="Model Output">


<script>
  document.addEventListener('DOMContentLoaded', (event) => {
    function initializeSlider(sliderId, outputImageId, sliderValueId, autoplayButtonId, folder) {
      const slider = document.getElementById(sliderId);
      const outputImage = document.getElementById(outputImageId);
      const sliderValue = document.getElementById(sliderValueId);
      const autoplayButton = document.getElementById(autoplayButtonId);

      let autoplay = true;
      let interval;

      function updateSlider(value) {
        sliderValue.textContent = `CFG scale ${value}`;
        outputImage.src = `/${folder}_slider/${folder}-CFG${value}_grid.png`;
      }

      slider.addEventListener('input', (event) => {
        const value = event.target.value;
        updateSlider(value);
      });

      autoplayButton.addEventListener('click', () => {
        autoplay = !autoplay;
        autoplayButton.textContent = autoplay ? 'Pause' : 'Play';
        if (autoplay) {
          startAutoplay();
        } else {
          clearInterval(interval);
        }
      });

      function startAutoplay() {
        interval = setInterval(() => {
          let value = parseInt(slider.value, 10);
          value = (value + 1) % 100;
          slider.value = value;
          updateSlider(value);
        }, 100); // Change the interval time as needed
      }

      // Start autoplay by default
      startAutoplay();
    }

    initializeSlider('parameterSlider1', 'outputImage1', 'sliderValue1', 'autoplayButton1', 'fire');
    initializeSlider('parameterSlider2', 'outputImage2', 'sliderValue2', 'autoplayButton2', 'bird');
    initializeSlider('parameterSlider3', 'outputImage3', 'sliderValue3', 'autoplayButton3', 'hair');
  });
</script>

<style>
  .slider-container {
    width: 80%; /* Adjust the width as a percentage of the window size */
    margin: 20px auto;
    display: flex; /* Use flexbox to align items horizontally */
    align-items: center; /* Center items vertically */
  }
  .parameterSlider {
    flex: 1; /* Allow the slider to grow and take available space */
    margin-left: 10px; /* Add some space between the value and the slider */
  }
  .outputImage {
    display: block;
    margin: 20px auto;
    max-width: 80%; /* Adjust the width of the image relative to the window size */
  }
  .sliderValue {
    min-width: 100px; /* Ensure the value box has some width */
    text-align: right; /* Align the text inside the value box to the right */
  }
  .autoplayButton {
    min-width: 60px; /* Set a fixed minimum width for the button */
    margin-right: 10px; /* Add some space between the button and the value */
  }
  body {
    margin: 0; /* Remove default margin */
    padding: 0; /* Remove default padding */
  }
</style>

