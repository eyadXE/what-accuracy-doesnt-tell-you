# Figure captions

**fig01_lossy_pipeline.png**
- Caption: The four stages of the pipeline and what each arrow throws away.
- Section: The pipeline and its three lossy arrows
- Alt text: Diagram showing raw logits flowing through softmax, argmax, and accuracy, with the information discarded at each step labeled underneath.

**fig02_logit_space_3d.png**
- Caption: Test-set logits from a 3-class MNIST model plotted in R^3, with the all-ones direction (1,1,1) drawn through their centroid. Sliding a point along this line (red dot -> triangle) changes nothing softmax can see.
- Section: What softmax can't see: the shift invariance
- Alt text: 3D scatter plot of logits for a 3-class digit classifier, colored by class, with a dashed line showing the all-ones direction and a red arrow showing a point shifted along it.

**fig03_probability_simplex.png**
- Caption: 3-class test samples plotted on the probability simplex. Correct predictions cluster at the vertices; most errors sit near an edge, where two classes are genuinely contested.
- Section: The probability simplex (triangle)
- Alt text: Triangle (2-simplex) scatter plot of softmax outputs for a 3-class model, blue dots for correct predictions piled at the corners, red x marks for incorrect predictions mostly along the edges.

**fig04_temperature_simplex_trajectories.png**
- Caption: Six real samples' softmax outputs as alpha sweeps from near 0 to 8 (z' = alpha*z). Each trajectory starts at the uniform center and slides toward a vertex, but never crosses into a different decision region.
- Section: Turning the temperature knob: same accuracy, different truth
- Alt text: Triangle simplex plot showing six colored trajectories moving from the center toward the corners as a scaling factor alpha increases, none crossing into a different third of the triangle.

**fig05_margin_geometry_2d.png**
- Caption: Top-2 logits ($z_1$ vs $z_2$) for E_cifar10_resnet18 test samples. The dashed line is the decision boundary $z_1=z_2$; the margin is each point's distance to it.
- Section: Margin vs confidence: the plot twist
- Alt text: Scatter plot of the top logit versus the second logit, with a diagonal decision boundary line; correct predictions in blue sit off the line, incorrect predictions in red cluster near it.

**fig06_sigmoid_of_margin.png**
- Caption: The sigmoid of the logit margin, with real samples overlaid. Confidence saturates quickly: most of the margin axis maps to a probability indistinguishable from 1.
- Section: Turning the temperature knob: same accuracy, different truth
- Alt text: Sigmoid curve of confidence versus logit margin with scattered data points, flattening near 1.0 well before the margin reaches 10.

**fig07_reliability_diagrams.png**
- Caption: Reliability diagrams for F_cifar100_resnet18 before and after temperature scaling (T=2.47). The shaded gap between the accuracy bar and the diagonal is the overconfidence gap; the histogram below shows how many test samples fall in each confidence bin.
- Section: Does 90% mean 90%? A calibration reality check
- Alt text: Two reliability diagrams side by side, before and after temperature scaling, each with a bar chart of accuracy per confidence bin against the diagonal, and a count histogram underneath.

**fig08_accuracy_vs_nll_overtraining.png**
- Caption: Validation accuracy and validation NLL over training for B_mnist_mlp_overtrained. Accuracy saturates early while NLL keeps climbing: the model is getting more confidently wrong on the mistakes it still makes, even as its scoreboard result stops changing.
- Section: Does 90% mean 90%? A calibration reality check
- Alt text: Line chart with two y-axes: validation accuracy flattening early in blue, validation NLL rising over later epochs in red, versus training epoch on the x-axis.

**fig09_id_vs_ood_softmax_vs_energy.png**
- Caption: ID (D_cifar10_smallcnn) vs OOD (gaussian) score distributions. Both scores overlap heavily with ID here — and on energy specifically, the noise cluster sits at *higher* (more ID-like) values than the real test images, giving an OOD-AUROC of 0.476, below chance. The energy-beats-softmax story from Liu et al. doesn't hold for every model.
- Section: Confident on garbage: OOD and the energy that softmax threw away
- Alt text: Two overlaid histograms comparing in-distribution and out-of-distribution samples: left panel by max softmax probability shows heavy overlap, right panel by energy score shows the noise cluster shifted to look even more in-distribution than the real photos.

**fig10_information_loss_heatmap.png**
- Caption: AUROC for detecting misclassifications, by score (pipeline stage) and model. Darker cells (closer to 1.0) mean that score ranks correct vs incorrect predictions well; argmax-only is a flat 0.5 by construction.
- Section: The scoreboard: how much each stage loses
- Alt text: Heatmap of AUROC values with models on the rows and scoring functions on the columns, darker blue indicating higher AUROC, numeric value in each cell.

**fig11_confidently_wrong_gallery.png**
- Caption: The 16 highest-confidence mistakes for E_cifar10_resnet18: true label, predicted label, and the model's confidence for each.
- Section: Hall of fame: the most confidently wrong predictions
- Alt text: Grid of small images, each captioned with the true label, the predicted label, and the model's confidence, showing the model's most confident mistakes.
