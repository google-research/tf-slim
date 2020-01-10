# Contributing

Interested in contributing to TF-Slim? We appreciate your interest but at this
point we cannot accept new contributions, only bug fixes.

## Pull Requests

We gladly welcome [pull requests](
https://help.github.com/articles/about-pull-requests/).

Before making any changes, we recommend opening an issue (if it
doesn't already exist) and discussing your proposed changes. This will
let us give you advice on the proposed changes. If the changes are
minor, then feel free to make them without discussion.


All submissions, including submissions by project members, require review. After
a pull request is approved, we merge it. Note our merging process differs
from GitHub in that we pull and submit the change into an internal version
control system. This system automatically pushes a git commit to the GitHub
repository (with credit to the original author) and closes the pull request.



## Unit tests

All TF-Slim code-paths must be unit-tested.  See existing unit tests
for recommended test setup.

Unit tests ensure new features (a) work correctly and (b) guard against future
breaking changes (thus lower maintenance costs).

To run existing unit-tests, use the command:


```shell
python setup.py test
```

from the root of the `tf_slim` repository, ideally inside a virtualenv.
The tests will run with CPU or GPU, depending on which version of TensorFlow
you have installed.


## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.
