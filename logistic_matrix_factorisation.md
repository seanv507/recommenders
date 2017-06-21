review of Logistic matrix factorisation for Implicit feedback data
================
Sean Violante (original author Christopher C Johnson @ spotify)

*U* = (*u*<sub>1</sub>, …, *u*<sub>*n*</sub>) *I* = (*i*<sub>1</sub>, …, *i*<sub>*m*</sub>) $R = (R\_{ui})\_{n\\times m},\\ \\mathbb R\_{\\ge 0}$

$$p(l\_{ui} | x\_u, y\_i, \\beta\_u, \\beta\_i) = \\frac{\\exp(x\_u y\_i^T+\\beta\_u+\\beta\_i)}{1 + \\exp(x\_u y\_i^T +\\beta\_u+\\beta\_i)}
$$

*β*<sub>*u*</sub> is a user bias - how likely to play new music? *β*<sub>*i*</sub> is a item bias - how likely new user to play music?

Define confidence in entries, as *c* = *α**r*<sub>*u**i*</sub>, can also use *c* = 1 + *α**l**o**g*(1 + *r*<sub>*u**i*</sub>) to damp down power users. Treat each zero observation as single observation of negative observation and *r*<sub>*u**i*</sub> be c positive classes.

Make assumption that all entries are conditionally independent:
$$ \\mathsf L (R|X,Y,\\beta) =\\prod\_{u,i}p(l\_{ui} | x\_u, y\_i, \\beta\_u, \\beta\_i)^{\\alpha r\_{ui}}(1-p(l\_{ui} | x\_u, y\_i, \\beta\_u, \\beta\_i))$$

SEAN: This seems wrong, but irrelevant? (on assumption *c* ≫ 1). case *r*<sub>*u**i*</sub> = 0 correct, but case *r*<sub>*u**i*</sub> ≠ 0 should not have negative instance case.
$$ \\mathcal L (R|X,Y,\\beta) =\\prod\_{u,i}p(l\_{ui} | x\_u, y\_i, \\beta\_u, \\beta\_i)^{\\alpha r\_{ui}}(1-p(l\_{ui} | x\_u, y\_i, \\beta\_u, \\beta\_i))$$

$$ \\log p (X,Y,\\beta |R) =\\sum\_{u,i} \\alpha r\_{ui} (x\_u y\_i^T + \\beta\_u + \\beta\_i) 
- (1 + \\alpha r\_{ui}) \\log(1 + \\exp(x\_u y\_i^T +\\beta\_u+\\beta\_i)) -\\frac{\\lambda}{2}\\|x\_u\\|^2 - \\frac{\\lambda}{2}\\|y\_i\\|^2$$

should rather be
$$ \\log p (X,Y,\\beta |R) =\\sum\_{u,i}  - \\alpha r\_{ui} \\log(1 + \\exp( - (x\_u y\_i^T +\\beta\_u+\\beta\_i))
- \\mathbb{1}\_{r\_{ui}=0} \\log(1 + \\exp(  (x\_u y\_i^T +\\beta\_u+\\beta\_i)) 
-\\frac{\\lambda}{2}\\|x\_u\\|^2 - \\frac{\\lambda}{2}\\|y\_i\\|^2$$

Alternate Gradient ascent for user vectors and biases, then item vectors and biases

Each iteration linear in users and items
