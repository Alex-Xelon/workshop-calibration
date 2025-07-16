# AI Model Calibration Workshop – Building Trustworthy Predictions

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.13+-blue)
![Marimo Notebook](https://img.shields.io/badge/marimo-notebook-blue)
![scikit-learn](https://img.shields.io/badge/library-scikit--learn-blue)
![GitHub repo size](https://img.shields.io/github/repo-size/alexlemiere/calibration-test)
![Last Commit](https://img.shields.io/github/last-commit/alexlemiere/calibration-test)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)


This repository is designed to help you learn about **AI Model Calibration** - a crucial aspect of building trustworthy machine learning systems. Through hands-on exercises, you'll learn how to assess and improve the reliability of your model's confidence scores, ensuring that predicted probabilities actually correspond to real-world accuracies.

---

## Table of Contents
- **[Workshop Goals](#workshop-goals)**
- **[Branch Organization](#branch-organization)**
- **[Project Structure](#project-structure)**
- **[Setup](#setup)**

---

## Workshop Goals

- Understand the importance of **model calibration** in AI systems
- Learn different calibration techniques (Platt Scaling, Isotonic Regression, etc.)
- Practice implementing calibration on real datasets
- Evaluate calibration quality using reliability diagrams and metrics
- Build confidence in your model's probability estimates

## Branch Organization

| Branch         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `main`         | Introduction to model calibration with theory and basic examples           |
| `easy`         | Guided version: pre-processed data and scaffolding provided               |
| `intermediate` | Moderate guidance: implement calibration from scratch with some hints      |
| `hard`         | Advanced version: tackle real-world calibration challenges independently   |
| `correction`   | Complete solution with detailed explanations and best practices            |

> Start with the `main` branch to understand the theoretical foundations. Choose your challenge level based on your experience with machine learning and probability theory. Feel free to start with the `hard` branch if you're up for a challenge. If it gets tricky, switch to `intermediate` or `easy` for progressive hints.

## Project Structure

The workshop is organized around progressive tutorials that build your understanding of model calibration, it designed to suit learners of all levels. We recommend starting with the `main` branch to understand the theoretical foundations before moving to other branches based on your desired challenge level.
The workshop offers progressive tutorials on model calibration, suitable for all skill levels.

Each branch (`easy`, `intermediate`, `hard`) includes:

- **Three progressive stages**:
  - `stage_1`: Basic binary classification calibration
  - `stage_2`: Multi-class calibration
  - `stage_3`: Multi-labels calibration

- **Standard directory structure**:
  - `data/`: Training and validation datasets
  - `src/calibration/`: Interactive notebooks with tutorials and examples for calibration implementations
  - `plots/`: Generated visualizations including reliability diagrams and calibration curves

- A `TODO.md` file at the root of the project provides instructions for each stage.

The `correction` branch contains complete implementations and detailed explanations.

To switch between versions of the workshop :
```bash
git checkout <branch-name>
```

Replace `<branch-name>` with one of the following :

- `main`         → basic explanation and introduction
- `easy`         → simplified version
- `intermediate` → default level
- `hard`         → advanced version
- `correction`   → full solution

---

## Setup

You can complete this workshop in **two ways** :

### Option 1 — Local setup (with VScode or Cursor)

If you're comfortable working locally, simply clone the repository and follow the common instructions above.

#### Common Instructions

Install `uv` (Python package manager with virtual environment support):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

>If the uv installation fails or you're having trouble with your virtual environment, you can try running the workshop in the cloud using GitHub Codespaces instead.

Then, run `uv sync` to install dependencies

```bash
uv sync
```

Finally, activate your environment using:

```bash
source .venv/bin/activate
```

### Option 2 — Run in GitHub Codespaces (cloud)

If you prefer not to set up anything locally, you can run everything directly in the browser via **GitHub Codespaces** — no installation required.

1. Go to the repository on GitHub.
2. **Select the branch you want to work on.**
3. Click on the green **Code** button → "Create codespace on `Branch's name` "
4. In the terminal that opens in the Codespace, run the following commands to set up your environment:
   ```bash
   uv sync
   source .venv/bin/activate
   ```
   You're now ready to start coding!

![illustration](assets/capture_1.png)

**Important** : Codespaces are tied to the branch selected when you create them.
If you want to switch branches later, go back to GitHub, select the new branch, and click on the + (plus) icon in the top right corner to create a new Codespace for that branch.
You will then need to repeat the terminal commands inside the new Codespace.

![illustration](assets/capture_2.png)

---

*You're now all set — pick your branch, open the exercise, and start exploring model calibration!*
