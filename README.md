# MindDiffuser

## Inspiration
In the world of Generative AI, content creation is one of the fastest-growing fields. While vast contents are increasing, our attention spans have decreased significantly. The growing need for personalized, engaging content in today's digital era inspired us to create MindDiffuser. We thought, what if we can convert neural data into personalized videos harnessing the power of Generative AI? This product can revolutionize the whole content creation and marketing world.

## What it does
MindDiffuser AI records your EEG data while watching a video, analyzes the moments in the video where you paid the most attention, and then turns those detected Salient EEG data into videos. It uses Generative AI Models to create videos from EEG data.

## How we built it
We can divide this part into 4 parts: Neuroscience, Frontend, Backend, and AI.

- **Neuroscience**: We collected EEG data with OpenBCI Cyton Board while a participant is looking at a video in a controlled environment. Then we do filtering, Artifact Removal (ICA), and frequency band analysis with MNE. For the Attention detection algorithm, we used the Alpha/Beta band ratio to compute high attention moments in the video.
- **AI**: EEG signals are normalized, padded to 128 channels, tokenized into fixed time windows, and embedded into high-dimensional representations. The model is pretrained to reconstruct masked portions of EEG embeddings using a reconstruction loss, learning global context. Latent EEG embeddings condition a diffusion-based generative model to map EEG signals to corresponding visual representations. The diffusion process iteratively refines noise into a coherent image aligned with the input EEG signals. The reconstructed images are sent to the Kling AI to output a video.
- **Frontend and Backend**: We use industry-standard React, Django, SQLite, and REST frameworks.

## Challenges we ran into
- **Noise in EEG data**: We had to figure out both online and offline filtering for pre-processing the data.
- **Transforming EEG data**: Transforming the EEG data to be used for our AI Models to generate content was challenging.
- **Generating Images with AI models**: Adapting the EEG data format and doing interpolation for models that can accept our EEG Data was difficult given the time frame.

## Accomplishments that we're proud of
Given the ambitious idea and timeframe, the fact that we can generate Images and Videos from EEG Data that is close to what a person saw is our biggest accomplishment. We believe that a proof of concept is a big accomplishment from the perspective of business validation. Doing something that has never been done from a product perspective in NeuroTech and MindDiffuser can create a revolution in the market. We also felt complete working in a diverse team with such dynamic and brilliant minds where learning and collaboration were core motivators for us.

## What we learned
- **Neuroscience**: Operationalizing attention, studying attention from EEG Alpha Beta Wave, filtering and pre-processing EEG data, and understanding how the OpenBCI Cyton board works.
- **Computing Science**: Understanding how AI models like Stable Diffusion work, implementing cutting-edge research papers on GenAI, clip encoding, grid generation for working with images in Generative AI models, Frontend and Backend integration in a production environment, Training AI Models with EEG data, and inferencing.
- **Product Development**: Frontend and specifically UI/UX is as important as Backend, prioritizing product features given time constraints.

## What's next for MindDiffuser
The next steps for MindDiffuser would be gaining traction and raising funds. We plan to reach out to incubators and validate our business models so that we are ready for the market. With funding from angel investors, we plan to improve our models so that it is trained on a larger dataset, invest in hardware, and build more features like Real-time insights, expanding capabilities to detect user moods for deeper personalization, and many more. We also plan to reach out to educational institutions for research partnership opportunities and collaborate with marketing industries. According to our Go To Market Strategy, we plan to begin awareness in Quarter 1 and do a full market entry within Quarter 3.

## Source Code
- [src-diffuser](src-diffuser)
- [src-django](src-django)
- [src-eeg-emissary](src-eeg-emissary)
- [src-vite](src-vite)
- [.gitignore](.gitignore)
- [README.md](README.md)
- [requirements.txt](requirements.txt)

## Database Setup
```sh
python3 src/manage.py makemigrations && python3 src/manage.py migrate
```
## Run the Application
```sh
python3 src/manage.py runserver
```
## Frontend Setup
This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type aware lint rules:

- Configure the top-level `parserOptions` property like this:

```js
export default tseslint.config({
  languageOptions: {
    // other options...
    parserOptions: {
      project: ['./tsconfig.node.json', './tsconfig.app.json'],
      tsconfigRootDir: import.meta.dirname,
    },
  },
})
```

- Replace `tseslint.configs.recommended` to `tseslint.configs.recommendedTypeChecked` or `tseslint.configs.strictTypeChecked`
- Optionally add `...tseslint.configs.stylisticTypeChecked`
- Install [eslint-plugin-react](https://github.com/jsx-eslint/eslint-plugin-react) and update the config:

```js
// eslint.config.js
import react from 'eslint-plugin-react'

export default tseslint.config({
  // Set the react version
