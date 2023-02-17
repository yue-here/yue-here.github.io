# Chemical Diffusion

Generative text-to-image models have recent become very popular. Having a bunch of leftover data from the [This JACS Does Not Exist](http://thisjacsdoesnotexist.com/) project, I've trained a Stable Diffusion checkpoint on the ~60K JACS table-of-contents images with paper titles as the captions. Here are some examples with the prompt in the caption:

<center>

![Development of a Highly Efficient and Selective Catalytic Enantioselective Hydrogenation for Organic Synthesis](/chemdiff_1.png)
<br>"Development of a Highly Efficient and Selective Catalytic Enantioselective Hydrogenation for Organic Synthesis"
<br><br>

![Lead-free Cs2AgBiBr6 Perovskite Solar Cells with High Efficiency and Stability](/chemdiff_2.png)
<br>"Lead-free Cs2AgBiBr6 Perovskite Solar Cells with High Efficiency and Stability"
<br><br>

![A Triazine-Based Covalent Organic Framework for High-Efficiency CO2 Capture](/chemdiff_3.png)
<br>"A Triazine-Based Covalent Organic Framework for High-Efficiency CO2 Capture"
<br><br>

![The Design and Synthesis of a New Family of Small Molecule Inhibitors Targeting the BCL-2 Protein](/chemdiff_4.png)
<br>"The Design and Synthesis of a New Family of Small Molecule Inhibitors Targeting the BCL-2 Protein"
<br><br>


</center>

## Running the model
The fun of generative models is in running it yourself of course. If you're not familiar with the process, here's a quick guide:
1. Install a Stable Diffusion UI. I've been using [this one](https://github.com/AUTOMATIC1111/stable-diffusion-webui), which has good installation instructions and works on both windows with NVIDIA/AMD GPUs and apple silicon.
2. Download the trained chemical diffusion checkpoint [hosted here on hugging face](https://huggingface.co/yuewu/chemical-diffusion/tree/main) - you just need to put the .ckpt file (~2.5GB) in the `\stable-diffusion-webui\models\Stable-diffusion` folder
3. Run the UI and have fun!

## Notes
Taking a page from the larger Stable Diffusion community, negative prompts can clean up the generated images - I've used 'out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts'.

Different samplers can have a big effect as well. Here's a grid showing the same prompt with several different samplers:
<center>

![Grid of samplers and sampling steps](/chemdiff_5.png)
<br>"The Discovery of a Highly Efficient and Stable Iridium Catalyst for the Oxygen Reduction Reaction in Fuel Cells"

</center>

## Training
As I'm VRAM-poor (running on 8GB 3060Ti), recent optimizations have made it possible to finetune SD on my home system where a few months ago twice as much memory was needed. I used the training UI [this repo](https://github.com/bmaltais/kohya_ss) which made the process very easy. YMMV - I ended up being able to train with batch sizes of 4, but frequently encountered the dreaded `CUDA out of memory` error.
