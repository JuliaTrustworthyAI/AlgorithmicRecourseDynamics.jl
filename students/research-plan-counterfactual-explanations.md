Responsible Professor: Dr. Cynthia Liem Supervisor: Patrick Altmeyer

## Research questions

Collectively, students will discuss and agree on an answer to the following question:

-   How can we quantify endogenous domain and model shifts?

Individually, each of the students will then choose one of the recourse generators already implemented in [CARLA](https://github.com/carla-recourse/CARLA) and run a simple experiment (details below) to generate endogenous shifts and quantify their magnitude. You will also each produce results for the baseline generator ([Wachter](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)) and benchmark the results from both generators.

Use the results to answer the following questions with respect to your chosen generator:

-   Does the magnitude of induced model shifts differ compared to the baseline generator?
-   If so, what factors might be playing a role here?
-   Based on your findings, what appear to be good ways to mitigate endogenous shifts?

While the individual part should really be done individually, do keep in mind that you will end up comparing the results of your experiments and you should therefore all generate the same, comparable output.

## Research Methods

Overall, this will be a project focused on benchmarking and experimental design: you will learn about how to use existing tools for benchmarking to asses outcomes from your own experiments.

For a gentle introduction to the topic of algorithmic recourse you can check out this [primer](https://towardsdatascience.com/individual-recourse-for-black-box-models-5e9ed1e4b4cc) written by Patrick as part of his PhD application. To familiarize yourself further with the topic and [CARLA](https://github.com/carla-recourse/CARLA) you should all read this [paper](https://arxiv.org/pdf/2108.00783.pdf).

You are free to choose any of the generators already implemented [CARLA](https://github.com/carla-recourse/CARLA), but below are five suggestions:

1.  [FACE](https://arxiv.org/pdf/1909.09369.pdf), \[1\]
2.  [REVISE](https://arxiv.org/pdf/1907.09615.pdf), \[2\]
3.  [CEM](https://arxiv.org/pdf/1802.07623.pdf), \[3\]
4.  [CEM-VAE](https://arxiv.org/pdf/1802.07623.pdf), \[3\]
5.  [AR-LIME](https://arxiv.org/pdf/1809.06514.pdf), \[4\]

Since the generators are all conveniently implemented in the same library, running the experiment should require roughly the same effort for all of them. While you are not expected to fully understand in detail the methodology underlying your chosen generator, you should study the relevant paper and ask yourself: what makes your generator particular? What assumptions do the authors make? How do you expect these particularities to affect recourse outcome with respect to endogenous shifts?

**Experiment**

The idea is that you replicate a variation of the following experiement already implemented by Patrick:

1.  Train an algorithm ![\\mathcal{M}](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BM%7D "\mathcal{M}") for a binary classification task.
2.  Determine a target class. Using your chosen generator and the baseline generator ([Wachter](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)) generate recourse for a share ![\\mu](https://latex.codecogs.com/svg.latex?%5Cmu "\mu") of randomly selected individuals in the other class to revise their label (i.e. move to the target class). Store some of the conventional benchmark measures readily implemented in CARLA (cost, success rate, …)
3.  Implement recourse for those indviduals and quantify the domain shift.
4.  Retrain ![\\mathcal{M}](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BM%7D "\mathcal{M}") and quantify the model shift.
5.  Repeat steps 1-4 for ![K](https://latex.codecogs.com/svg.latex?K "K") rounds.

You are free to either follow Patrick’s recipe exactly or tweak the various parameters that determine the experiemntal design as you see fit. Make sure that you end up running the exact same experiment though, in order for the results to be comparable.

**Hypothesis**

I expect to find that generators that focus on creating realistic counterfactuals in high-density regions of the feature domain (REVISE, CEM-VAE) tend to be associated with endogenous shifts of a smaller magnitude compared to generators that do not specifically address this (CEM, Wachter, AR-LIME).

## Relation between sub-projects

The students will collectively design and establish an evaluation benchmark, while each focusing on a particular framework implementation to ultimately evaluate in the benchmark. Researching the intricacies of each framework will lead to clearly individual contributions, while part of the relevant literature is shared, and the final comparison connects the individual insights.

## Responsible Engineering / Computer Science / Research

As noted above, algorithmic recourse was designed in the context of responsible machine learning. Work on this topic should foster an awareness that automated decision-making can have detrimental consequences for individuals and that many researchers are working on approaches to deal with some of those consequences. Comparing different approaches to the issue will also make it clear that in the context of research there is rarely (if ever) one right solution. More to the point of this particular research project, solutions to one particular issue usually open up new questions.

## Possible publication

In 2021 there was an [ICML workshop](https://icml.cc/Conferences/2021/ScheduleMultitrack?event=8363) on algorithmic recourse. Should this workshop take place again, the results of this project could lead to a poster presentation. Later in the year NeurIPS is coming up, which in 2021 also attracted conference and workshop papers on algorithmic recourse and benchmarking.

## References

<div id="refs" class="references csl-bib-body">

<div id="ref-poyiadzi2020face" class="csl-entry">

<span class="csl-left-margin">\[1\] </span><span class="csl-right-inline">Poyiadzi R, Sokol K, Santos-Rodriguez R, De Bie T, Flach P. FACE: Feasible and actionable counterfactual explanations. Proceedings of the AAAI/ACM conference on AI, ethics, and society, 2020, p. 344–50.</span>

</div>

<div id="ref-joshi2019towards" class="csl-entry">

<span class="csl-left-margin">\[2\] </span><span class="csl-right-inline">Joshi S, Koyejo O, Vijitbenjaronk W, Kim B, Ghosh J. Towards realistic individual recourse and actionable explanations in black-box decision making systems. arXiv Preprint arXiv:190709615 2019.</span>

</div>

<div id="ref-dhurandhar2018explanations" class="csl-entry">

<span class="csl-left-margin">\[3\] </span><span class="csl-right-inline">Dhurandhar A, Chen P-Y, Luss R, Tu C-C, Ting P, Shanmugam K, et al. Explanations based on the missing: Towards contrastive explanations with pertinent negatives. Advances in Neural Information Processing Systems 2018;31.</span>

</div>

<div id="ref-ustun2019actionable" class="csl-entry">

<span class="csl-left-margin">\[4\] </span><span class="csl-right-inline">Ustun B, Spangher A, Liu Y. Actionable recourse in linear classification. Proceedings of the conference on fairness, accountability, and transparency, 2019, p. 10–9.</span>

</div>

</div>
