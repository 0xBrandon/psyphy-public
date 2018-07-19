## PsyPhy: A Psychophysics Driven Evaluation Framework

Psyphy is a large-scale, flexible evaluation framework for performing psychophysics evaluation experiments on machine learning algorithms. At the moment, PsyPhy only has functionality for object classification and face identification but is easily extendable to almost any machine learning algorithm in computer vision, natural language processing, etc.

### Terms of Use

MIT License, see `LICENSE`.


### Installation

This will be available as a pip package in the near future, but for now feel free to clone the repository and use it locally.


### Description

Every psychophysics evaluation experiment is broken down into three components represented as three functions. The simplest component, called a perturbation function, is often defined and then reused in many experiment. Each might have a different decision function and item-response curve generator function, depending on the needs of the task (e.g. classification, matching, or detection).

* **_Perturbation Function_**

   A perturbation function is a function that takes as input a single input stimuli and a single perturbation value. It returns a single new stimuli that is the result of transforming the input stimuli by the amount represented by the perturbation value. Perturbation functions don't have to be used in conjunction with decision functions or item-response curve generators as they are stand-alone stimuli processing functions with input-output stimuli.

   **For complete detail**, see original publication describing the framework, *PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition*, linked below.

   **For examples**, see the module named `psyphy.perturb` which contains many pre-written perturbation functions

* **_Decision Function_**

   A decision function is an abstract definition of a model to be evaluated. This can be defined by an experimenter to be almost anything to suit the task desired. A decision function is tied to an item-response curve generator function in that the item-response curve generator function must know how to properly use the defined decision function.

   **For complete detail**, see original publication describing the framework, *PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition*, linked below.

   **For examples**, see the folder named `publications` where the source code for the listed publications lies.

* **_Item-Response Curve (IRC) Generator Function_**

  An item-response curve generator uses a decision function and a given perturbation function to produce an item-response curve, in the form of raw data that can be used for plotting with any plotting software. An item-response curve generator must know how to use the abstract decision function defined for the task at hand.

   **For complete detail**, see original publication describing the framework, *PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition*, linked below.

   **For examples**, see the module named `psyphy.n2nmatch` where the item-response curve generator functions are from *
Visual Psychophysics for Making Face Recognition Algorithms More Explainable*, liked below. See the module named `psyphy.classification` for the item-response curve generator function in *PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition*, also linked below.

### Tutorial

There will be a tutorial in the near future, but for now please see the examples in `publications` folder.


### Publications

(2018) [Visual Psychophysics for Making Face Recognition Algorithms More Explainable](http://www.bjrichardwebster.com/papers/menagerie/pdf) by Brandon RichardWebster, So Yon Kwon, Christopher Clarizio, Samuel E. Anthony, and Walter J. Scheirer.

(2018) [PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition](http://www.bjrichardwebster.com/papers/psyphy/pdf) by Brandon RichardWebster, Samuel E. Anthony, and Walter J. Scheirer.

### Citing

`@ARTICLE{8395028,
author={B. RichardWebster and S. Anthony and W. Scheirer},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
title={PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition},
year={2018},
volume={},
number={},
pages={1-1},
keywords={Computational modeling;Computer vision;Machine learning;Observers;Psychology;Task analysis;Visualization;Deep Learning;Evaluation;Neuroscience;Object Recognition;Psychology;Visual Psychophysics},
doi={10.1109/TPAMI.2018.2849989},
ISSN={0162-8828},
month={},}`
