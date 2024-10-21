<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/brunomaso1/uba-ceia">
    <img src="ceia-nlp/resources/logoFIUBA.jpg" alt="Logo" width="500">
  </a>

<h3 align="center">NLP</h3>

  <p align="center">
    Natural Language Processing
    <br />
    <a href="https://github.com/brunomaso1/uba-ceia"><strong>Explore the whole project »</strong></a>
    <br />
    <br />
    <a href="https://nbviewer.org/github/brunomaso1/uba-ceia/blob/ceia-nlp/ceia-nlp/Desafio%201.ipynb">View Demo</a>
    ·
    <a href="https://github.com/brunomaso1/uba-ceia/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/brunomaso1/uba-ceia/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Este proyecto se basa en el contenido de la materia NLP (Procesamiento del Lenguaje Natural) dentro del contexto de la carrera de Especialización en Inteligencia Artificial.

Consta de varios desafíos, pasando por varios temas, desde el concepto de embeddings hasta la generación de un modelo del lenguaje.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<div style="text-align: center">

[![Pytorch][pytorch]][pytorch-url]
[![Tensorflow][tensorflow]][tensorflow-url]
[![Jupyter][jupyter]][jupyter-url]
[![scikitlearn][scikitlearn]][scikitlearn-url]
[![pandas][pandas]][pandas-url]
[![keras][keras]][keras-url]
[![scipy][scipy]][scipy-url]

[![transformers][transformers]][transformers-url]
[![nltk][nltk]][nltk-url]
[![gensim][gensim]][gensim-url]
[![seaborn][seaborn]][seaborn-url]

</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Para la ejecución de los desafíos en tu ambiente local, sigue los siguientes pasos:

### Prerequisites

Como pre-requsito debes tener instalado `python`|`git`|`vscode` y tener un manejo mínimo comandos de consola. El proyecto fue creado utilizando `python 3.12`. Para descargar `python` puedes seguir el siguiente enlace: [Descargar python][download-python-url]

### Installation

1. Clonar el repositorio
   ```sh
   git clone https://github.com/brunomaso1/uba-ceia/tree/ceia-nlp.git
   ```
2. Dentro de la carpeta _ceia-nlp_, instala los requerimientos:
   ```sh
   pip install requirements.txt
   ```
3. Abre cualquier desafío `*.ipynb` con `vscode` y ejecútalo.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

El proyecto consta de varios desafíos, en la siguiente sección puedes ver las temáticas:

<details>

<summary>Desafío 1</summary>

<em><p>

<strong>Word vectorizer + Naive bayes:</strong>

El primer desfío consta de explorar la vectorización de documentos, con un enfoque clásico (utilizando temas TF-IDF), para empezar a visualizar la similitud entre documentos de forma matemática, utilizando usualmente la similitud de coseno. También es una primera aproximación a un simple modelo del lenguaje como lo es <ins>Naive Bayes</ins>, donde se analiza sus puntos fuertes así también como sus puntos débiles en comparación con <ins>Complement NB</ins> para la clasificación de documentos.

</p></em>

</details>

<details>

<summary>Desafío 2</summary>

<em><p>

<strong>Embeddings con Gensim:</strong>

El segundo desfío toma un enfoque más complejo e introduce el térmio de <ins>embeddings</ins>, el cual es una mejora a la primera vectorización. En este sentido, se analizan nuevamente medidas de distancias en las distintas dimensiones de estos embeddings creados con la librería de Gensim. Se exploran analogías y visualizaciones en espacios de dimensiones reducidos, mediante técnicas como PCA o T-SNE, utilizando el lore del Señor de los Anillos.

</p></em>

</details>

<details>

<summary>Desafío 3</summary>

<em><p>

<strong>Modelo del lenguaje:</strong>

El tercer desafío introduce formalmente las primeras lineas de investigación en modelos del lenguaje utilizando "deep learning". Se nombran térmios como tokenización y corpus, explorando en profundidad el proceso de tokenización, el cual es importante para entender como funcionan los grandes modelos del lenguaje, que si bien utilizan una tokenización disitinta actualmente, el enfoque no ha cambiado. En este desafío, también se analizan estructuras que en su tiempo fueron estado del arte, como las celdas de Elman (unidades recurrentes), o capas LSTM/GRU.

</p></em>

</details>

<details>

<summary>Desafío 4</summary>

<em><p>

<strong>BOT QA:</strong>

En este cuarto desafío, se intenta resolver un problema específico: crear un bot QA basando en "deep learning", utilizando capas LSTM y embeddings. En este sentido, se analizan los clasicos problemas que tienen estas arquitecturas de aprendizaje profundo, donde el "overfitting" o el "vanishing gradients" son temas que siempre están presentes. También se obtienen conclusiones interesantes desde el punto de vista de utilzar fine-tuning o la potencia de los embeddings pre-entrenados.

</p></em>

</details>

<details>

<summary>Desafío 5</summary>

<em><p>

<strong>BERT Sentiment Analysis:</strong>

En el útimo desafío, se explora una linea para resolver un problema, que hasta ahora, se venía enfocando con modelos de aprendizaje automático "clásico". Desde este punto de vista, se utliza BERT como encoder para la generación de embedding que condensan información implícita del lenguaje, para luego realizar una búsqueda vectorial con estos vectores para clasificarlos en distintas clases. Inicialmente, se plante resolver el problema con el modelo base, para luego evidenciar las ventajas (como también el costo computacional) del fine-tuning para estos casos.

</p></em>

</details>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- ✅ [Desafío 1][desafio1-notebook-url]
- ✅ [Desafío 2][desafio2-notebook-url]
- ✅ [Desafío 3][desafio3-notebook-url]
- ✅ [Desafío 4][desafio4-notebook-url]
- ✅ [Desafío 5][desafio5-notebook-url]

Consulte los [problemas abiertos](https://github.com/brunomaso1/uba-ceia/issues) para obtener una lista completa de las funciones propuestas (y problemas conocidos).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Las contribuciones son las que hacen de la comunidad de código abierto un lugar increíble para aprender, inspirar y crear. Cualquier contribución que hagas será **muy apreciada**.

Si tienes alguna sugerencia que pueda mejorar esto, haga un "fork" del repositorio y cree una "pull request". También puedes simplemente abrir un problema con la etiqueta "mejora".
¡No olvides darle una estrella al proyecto! ¡Gracias de nuevo!

1. Haga "Fork" del proyecto
2. Cree su rama de funciones (`git checkout -b feature/AmazingFeature`)
3. Confirme sus cambios (`git commit -m 'Agregar alguna característica sorprendente'`)
4. Empuje a la rama (`git push origin feature/AmazingFeature`)
5. Abra una solicitud de cambios ("pr")

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/brunomaso1/uba-ceia/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=brunomaso1/uba-ceia" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Bruno Masoller - [@brunomaso1](https://github.com/brunomaso1) - brunomaso1@gmail.com

Link del proyecto: [https://github.com/brunomaso1/uba-ceia](https://github.com/brunomaso1/uba-ceia)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Rodrigo Cardenas](https://github.com/rodo-qatar): Por los conocimientos, material y tiempo compartido.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/brunomaso1/uba-ceia.svg?style=for-the-badge
[contributors-url]: https://github.com/brunomaso1/uba-ceia/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/brunomaso1/uba-ceia.svg?style=for-the-badge
[forks-url]: https://github.com/brunomaso1/uba-ceia/network/members
[stars-shield]: https://img.shields.io/github/stars/brunomaso1/uba-ceia.svg?style=for-the-badge
[stars-url]: https://github.com/brunomaso1/uba-ceia/stargazers
[issues-shield]: https://img.shields.io/github/issues/brunomaso1/uba-ceia.svg?style=for-the-badge
[issues-url]: https://github.com/brunomaso1/uba-ceia/issues
[license-shield]: https://img.shields.io/github/license/brunomaso1/uba-ceia.svg?style=for-the-badge
[license-url]: LICENCE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/brunomaso1
[product-screenshot]: ceia-nlp/resources/readme-portada.png
[download-python-url]: https://www.python.org/downloads/

[pytorch]: https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[tensorflow]: https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[tensorflow-url]: https://www.tensorflow.org
[jupyter]: https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[jupyter-url]: https://jupyter.org/
[scikitlearn]: https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white
[pandas]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[keras]: https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white
[scipy]: https://img.shields.io/badge/scipy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white
[scikitlearn-url]: https://scikit-learn.org/
[pandas-url]: https://pandas.pydata.org/
[keras-url]: https://keras.io/
[scipy-url]: https://scipy.org/
[transformers]: https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black
[transformers-url]: https://huggingface.co/docs/transformers/index
[nltk]: https://img.shields.io/badge/nltk-0d97ca?style=for-the-badge
[gensim]: https://img.shields.io/badge/gensim-1938c0?style=for-the-badge
[seaborn]: https://img.shields.io/badge/seaborn-008ef4?style=for-the-badge
[nltk-url]: https://www.nltk.org/
[gensim-url]: https://radimrehurek.com/gensim/
[seaborn-url]: https://seaborn.pydata.org/
[desafio1-notebook-url]: ceia-nlp/Desafio%201.ipynb
[desafio2-notebook-url]: ceia-nlp/Desafio%202.ipynb
[desafio3-notebook-url]: ceia-nlp/Desafio%203.ipynb
[desafio4-notebook-url]: ceia-nlp/Desafio%204.ipynb
[desafio5-notebook-url]: ceia-nlp/Desafio%205.ipynb
