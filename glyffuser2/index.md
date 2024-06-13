# Classifier-free guidance for the Glyffuser

_See the main Glyffuser article [here]({{< ref "glyffuser.md" >}})_

<center>
{{< figure src="/CFG grabber.png">}}

_Here we test classifier-free guidance (CFG) as a method to enhance adherance to text conditioning. For the prompt "walk" (Chinese character 走), as we increase the CFG multiplier we end up with a double "walk" character that still follows the rules of Chinese glyph construction_
</center>
<br>

### Intro
[Classifier-free guidance](https://arxiv.org/abs/2207.12598) is an elegant and powerful technique that has recently become ubiquitous in conditional diffusion models. (For an excellent treatment, see [here](https://sander.ai/2022/05/26/guidance.html))

Essentially, this method allows the strength of any given prompt to be varied without needing to perform any additional training. Moreover, the strength of the prompt can be increased far above that for standard conditional training.

To implement this method, we simply add random dropout of the text conditioning tokens during training (10-20% has been found to work well). This effectively trains an unconditional model at the same time. During sampling steps, we simply perform the noise prediction twice, once normally and once with a zero conditioning tensor. We then combine them as follows:

`noise_prediction = noise_prediction_unconditional + guidance_scale * (noise_prediction -  noise_prediction_unconditional)`

{{< admonition Note "Note">}}
At `guidance_scale = 0`, the model acts as an unconditional model while at `guidance_scale = 1`, the model acts as the standard conditional model
{{< /admonition >}}

### Testing CFG scales
Generally, increasing `guidance_scale` in text-to-image models decreases variety while increasing adherence to the prompt. Let's try probing the model by varying the number of sampling steps and guidance scale for the prompt "bird" corresponding to a very common radical (鳥/鸟):

<center>
{{< figure src="/cfg-steps grid.png">}}
</center>

{{< admonition Note "Note">}}
Unusually, the "bird" radical can occur on either the left ("鸵", ostrich) or right ("鸡"， chicken) sides of characters.
{{< /admonition >}}

Compared to [previously]({{< ref "glyffuser.md" >}}), we see that as we increase the guidance scale, the 'bird' radical becomes increasingly activated from the very first sampling step. Interestingly, while the traditional form of the bird character "鳥" dominates (it is more prevalent in the training set), the simplified form "鸟" also makes a single appearance (10 steps, scale=50), making it a 'transition state' during the denoising process. The explorer below shows CFG scales of 0 to 100 for different random seeds - higher CFG scales do indeed reduce sample variety. Compared to general-purpose text-to-image models however, we can tolerate higher CFG scales as they tend to give more convincing characters. If you follow any individual character, you'll see that it tends to start with one 'bird' radical, then as CFG scale increases, at some point the other side will also collapse to a 'bird' radical.

<!-- <center>
  <video width="100%" autoplay loop muted playsinline>
    <source src="/bird_CFG1-100_compressed.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</center> -->

<div class="slider-container">
  <span id="sliderValue">CFG scale 1</span>
  <input type="range" min="0" max="99" step="1" value="1" id="parameterSlider">
</div>
<img id="outputImage" src="/bird_slider/bird-CFG1_grid.png" alt="Model Output">

<script>
  const slider = document.getElementById('parameterSlider');
  const outputImage = document.getElementById('outputImage');
  const sliderValue = document.getElementById('sliderValue');

  slider.addEventListener('input', (event) => {
    const value = event.target.value; // Use the integer value directly
    sliderValue.textContent = `CFG scale ${value}`; // Update the displayed value
    outputImage.src = `/bird_slider/bird-CFG${value}_grid.png`;
  });
</script>

<style>
  .slider-container {
    width: 80%; /* Adjust the width as a percentage of the window size */
    margin: 20px auto;
    display: flex; /* Use flexbox to align items horizontally */
    align-items: center; /* Center items vertically */
  }
  input[type="range"] {
    flex: 1; /* Allow the slider to grow and take available space */
    margin-left: 10px; /* Add some space between the value and the slider */
  }
  img {
    display: block;
    margin: 20px auto;
    max-width: 80%; /* Adjust the width of the image relative to the window size */
  }
  #sliderValue {
    min-width: 100px; /* Ensure the value box has some width */
    text-align: right; /* Align the text inside the value box to the right */
  }
  body {
    margin: 0; /* Remove default margin */
    padding: 0; /* Remove default padding */
  }
</style>


### CFG generations for the most common radicals
For completeness, the effect of CFG on all of our previous generations is shown below. Only for radicals such as 'bird', 'fire' (火) and 'walk' (走) do we see multiples - these are the radicals which in known characters can lie on different sides.
<center>
{{< figure src="/guidance_scale_grid.png">}}
</center>

### Bonus: CFG variations for "fire"
The Chinese character for fire "火" has a particularly varied set of possible locations. These are showcased in the characters "炎" and "焱". Another form is the bottom radical "灬", a kind of deconstructed version of "火". As such, greater variety is possible and this shows:
<!-- <center>
  <video width="100%" autoplay loop muted playsinline>
    <source src="/fire_CFG1-100_compressed.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</center> -->

<div class="slider-container slider-container-2">
  <span id="sliderValue1" class="sliderValue">CFG scale 1</span>
  <input type="range" min="0" max="99" step="1" value="1" id="parameterSlider1" class="parameterSlider">
</div>
<img id="outputImage1" class="outputImage" src="/fire_slider/fire-CFG1_grid.png" alt="Model Output">

<script>
  document.addEventListener('DOMContentLoaded', (event) => {
    const slider1 = document.getElementById('parameterSlider1');
    const outputImage1 = document.getElementById('outputImage1');
    const sliderValue1 = document.getElementById('sliderValue1');

    slider1.addEventListener('input', (event) => {
      const value = event.target.value; // Use the integer value directly
      sliderValue1.textContent = `CFG scale ${value}`; // Update the displayed value
      outputImage1.src = `/fire_slider/fire-CFG${value}_grid.png`;
    });
  });
</script>

<style>
  .slider-container-2 {
    width: 80%; /* Adjust the width as a percentage of the window size */
    margin: 20px auto;
    display: flex; /* Use flexbox to align items horizontally */
    align-items: center; /* Center items vertically */
  }
  .slider-container-2 .parameterSlider {
    flex: 1; /* Allow the slider to grow and take available space */
    margin-left: 10px; /* Add some space between the value and the slider */
  }
  .slider-container-2 .outputImage {
    display: block;
    margin: 20px auto;
    max-width: 80%; /* Adjust the width of the image relative to the window size */
  }
  .slider-container-2 .sliderValue {
    min-width: 100px; /* Ensure the value box has some width */
    text-align: right; /* Align the text inside the value box to the right */
  }
  body {
    margin: 0; /* Remove default margin */
    padding: 0; /* Remove default padding */
  }
</style>

