# Byte-Latent embeddings learning emergent behavior

## About

Emergent behavior is a complex behavior that arises from local interactions that influence
the global state, termed as emergence. Such patterns arise in real-world scenarios such as
ant colonies, crowd dynamics, market dynamics and in studies of chaotic systems such as
cellular automata, Lorenz attractor and logistic map. The project will adopt an incremental
approach in training a byte-latent LSTM architecture that leverages the entropy of data as
the model learns from simpler emergence onto growing complexities, to able validation on
lower-dimensional environments. The research aims at contributing to understanding and
foreseeing the behavior of emergence with this transformer architecture.

## Description

Predicting emergence is a challenge due to various non-linearities and chaotic
interactions. SotA (State of the Art) models dealing with such challenges have specific
limitations, such as the Physics Informed Neural Networks rely on explicit differential
equations and fail at adapting to unknown behavior and capturing long-term predictions.
On the other hand, transformers have shown capabilities in such regard, but they interpret
on fixed tokenization but integrating latent space representations is yet to be explored.
The byte-latent transformer architecture aims to capture the local interactions and long-
range dependencies via dynamically generating latent tokens from byte-sequences. This
scheme would effectively group local information and preserve high-level representations.
Additionally for the benefit of reduced computational complexity, interpretability and
generalization, the model will train with sparse attention mechanisms. The sparse
attention would focus on important patterns in the chaotic system especially in dealing
with varying kinds of data such as biological or physical emergence.

## References
