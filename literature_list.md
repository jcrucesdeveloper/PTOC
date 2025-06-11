### [Generalizing Shape Analysis with Gradual Types](https://web.cs.ucla.edu/~palsberg/paper/ecoop24.pdf)

- Tensors are multi-dimensional data structures that can represent the data processed by machine11
  learning tasks. Tensor programs tend to be short and readable, and they can leverage libraries and12
  frameworks such as TensorFlow and PyTorch, as well as modern hardware such as GPUs and TPUs.13
  However, tensor programs also tend to obscure shape information, which can cause shape errors14
  that are difficult to find. Such shape errors can be avoided by a combination of shape annotations15
  and shape analysis, but such annotations are burdensome to come up with manually.16
  In this paper, we use gradual typing to reduce the barrier of entry. Gradual typing offers a way17
  to incrementally introduce type annotations into programs. From there, we focus on tool support18
  for type migration, which is a concept that closely models code-annotation tasks and allows us to do19
  shape reasoning and utilize it for different purposes. We develop a comprehensive gradual typing20
  theory to reason about tensor shapes. We then ask three fundamental questions about a gradually21
  typed tensor program. (1) Does the program have a static migration? (2) Given a program and22
  some arithmetic constraints on shapes, can we migrate the program according to the constraints?23
  (3) Can we eliminate branches that depend on shapes? We develop novel tools to address the three24
  problems. For the third problem, there are currently two PyTorch tools that aim to eliminate25
  branches. They do so by eliminating them for just a single input. Our tool is the first to eliminate26
  branches for an infinite class of inputs, using static shape information. Our tools help prevent bugs,27
  alleviate the burden on the programmer of annotating the program, and improves the process of28
  program transformation.

### Literature Review: Tensor Shape Mismatch Detection

- [Gradual Tensor Shape Checking (full version)[2023]](https://arxiv.org/pdf/2203.08402)

  Tensor shape mismatch is a common source of bugs in deep
  learning programs. We propose a new type-based approach to detect
  tensor shape mismatches. One of the main features of our approach is
  the best-effort shape inference. As the tensor shape inference problem
  is undecidable in general, we allow static type/shape inference to be
  performed only in a best-effort manner. If the static inference cannot
  guarantee the absence of the shape inconsistencies, dynamic checks are
  inserted into the program. Another main feature is gradual typing, where
  users can improve the precision of the inference by adding appropriate
  type annotations to the program. We formalize our approach and prove
  that it satisfies the criteria of gradual typing proposed by Siek et al.

- [A Static Analyzer for Detecting Tensor Shape Errors in Deep Neural Network Training Code [2021]](https://sf.snu.ac.kr/publications/pytea.pdf)

  We present an automatic static analyzer PyTea that detects tensorshape errors in PyTorch code. The tensor-shape error is critical in
  the deep neural net code; much of the training cost and intermediate results are to be lost once a tensor shape mismatch occurs in
  the midst of the training phase. Given the input PyTorch source,
  PyTea statically traces every possible execution path, collects tensor
  shape constraints required by the tensor operation sequence of the
  path, and decides if the constraints are unsatisfiable (hence a shape
  error can occur). PyTea’s scalability and precision hinges on the
  characteristics of real-world PyTorch applications: the number of
  execution paths after PyTea’s conservative pruning rarely explodes
  and loops are simple enough to be circumscribed by our symbolic
  abstraction. We tested PyTea against the projects in the official
  PyTorch repository and some tensor-error code questioned in the
  StackOverflow. PyTea successfully detects tensor shape errors in
  these codes, each within a few seconds.

- [Static Analysis of Shape in TensorFlow Programs [2020]](https://drops.dagstuhl.de/storage/00lipics/lipics-vol166-ecoop2020/LIPIcs.ECOOP.2020.15/LIPIcs.ECOOP.2020.15.pdf)

  Machine learning has been widely adopted in diverse science and engineering domains, aided by
  reusable libraries and quick development patterns. The TensorFlow library is probably the bestknown representative of this trend and most users employ the Python API to its powerful back-end.
  TensorFlow programs are susceptible to several systematic errors, especially in the dynamic typing
  setting of Python. We present Pythia, a static analysis that tracks the shapes of tensors across
  Python library calls and warns of several possible mismatches. The key technical aspects are a close
  modeling of library semantics with respect to tensor shape, and an identification of violations and
  error-prone patterns. Pythia is powerful enough to statically detect (with 84.62% precision) 11 of
  the 14 shape-related TensorFlow bugs in the recent Zhang et al. empirical study – an independent
  slice of real-world bugs

- [Ariadne: Analysis for Machine Learning Programs [2018]](https://arxiv.org/pdf/1805.04058)

  Machine learning has transformed domains like vision and
  translation, and is now increasingly used in science, where
  the correctness of such code is vital. Python is popular for
  machine learning, in part because of its wealth of machine
  learning libraries, and is felt to make development faster;
  however, this dynamic language has less support for error
  detection at code creation time than tools like Eclipse. This
  is especially problematic for machine learning: given its statistical nature, code with subtle errors may run and produce
  results that look plausible but are meaningless. This can
  vitiate scientific results. We report on Ariadne: applying a
  static framework, WALA, to machine learning code that uses
  TensorFlow. We have created static analysis for Python, a
  type system for tracking tensors—Tensorflow’s core data
  structures—and a data flow analysis to track their usage. We
  report on how it was built and present some early results

- [An Empirical Study on Tensor Shape Faults in Deep Learning Systems [2021]](https://arxiv.org/pdf/2106.02887)

  Software developers frequently adopt deep learning
  (DL) libraries to incorporate learning solutions into software
  systems. However, misuses of these libraries can cause various
  DL faults. Among them, tensor shape faults are most prevalent. Tensor shape faults occur when restriction conditions of
  operations are not met, leading to many system crashes. To
  support efficient detection and fixing of these faults, we conduct
  an empirical study to obtain a deep insight. We construct SFData,
  a set of 146 buggy programs with crashing tensor shape faults
  (i.e., those causing programs to crash). By analyzing the faults in
  SFData, we categorize them into four types and get some valuable
  observations.
  Index Terms

- [Using Run-Time Information to Enhance Static Analysis of Machine Learning Code in Notebooks [2024]](https://dl.acm.org/doi/pdf/10.1145/3663529.3663785)

  A prevalent method for developing machine learning (ML) prototypes involves the use of notebooks. Notebooks are sequences of
  cells containing both code and natural language documentation.
  When executed during development, these code cells provide valuable run-time information. Nevertheless, current static analyzers
  for notebooks do not leverage this run-time information to detect
  ML bugs. Consequently, our primary proposition in this paper is
  that harvesting this run-time information in notebooks can significantly improve the effectiveness of static analysis in detecting
  ML bugs. To substantiate our claim, we focus on bugs related to
  tensor shapes and conduct experiments using two static analyzers:

1. pythia, a traditional rule-based static analyzer, and 2) gpt-4,
   a large language model that can also be used as a static analyzer.
   The results demonstrate that using run-time information in static
   analyzers enhances their bug detection performance and it also
   helped reveal a hidden bug in a public dataset.

- [torchtyping ](https://github.com/patrick-kidger/torchtyping)
  Type annotations for a tensor's shape, dtype, names, ...
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
  Type annotations and runtime checking for shape and dtype of JAX/NumPy/PyTorch/etc. arrays. https://docs.kidger.site/jaxtyping/
