# Finding patterns in an opponent's games

by Danish Puri

This is the design behind `backend/patterns`. The short version is that a deep
network supplies the geometry and simple statistics do the rest, because the
amount of data I have on any one opponent is tiny and pretending otherwise is
the fastest way to produce confident nonsense.

## The problem with what I had

The opening tables in `backend/analysis.py` bucket games by ECO code. ECO labels
a move order and stops around move ten, which causes two problems I kept hitting
when preparing.

Transpositions get filed apart. The same middlegame reached through the English
and through the Nimzo lands in two different rows, each looking like a small
sample, when really it is one recurring position type with twice the evidence.

And the middlegame is not covered at all. If someone handles isolated queen pawn
positions badly, that is worth knowing regardless of which opening produced one.
No ECO code can express it.

I wanted patterns keyed on positions rather than on names.

## Why the network never sees the opponent

Three hundred games is three hundred samples. Any model with real capacity fitted
to that will memorise it, and I would end up reading my own training noise back
as insight.

So the deep model is pretrained on a corpus of other people's games and frozen
before it ever touches the player I am preparing for. Everything per opponent is
non parametric, meaning clustering, retrieval and counting, with no parameters
fitted beyond cluster centroids. The network contributes a coordinate system.
The opponent's games only ever get counted inside it.

This is the standard frozen encoder pattern and it is the only statistically
honest way I could find to use deep learning at this sample size.

## The pretext task

The network predicts which move a human played, factored into a from square and
a to square, trained with cross entropy on both. The layer under that prediction
is the embedding.

The choice of pretext task is doing real work here. A network trained to predict
human choices has to represent what a position is asking of the player, which is
exactly the axis I want positions to cluster along. Something trained to predict
material or game result would organise the space by who is winning, which I do
not care about. I want positions that pose the same problem to sit together.

The factored head is an approximation, since it cannot express the joint
distribution over from and to squares. I renormalise over legal moves only at
inference, which repairs most of it, and the alternative full 4096 way head costs
several times the parameters for a signal I only use as a summary statistic.

## The pipeline

Everything before the encoder exists to hand it fewer positions. That is the
whole performance story, because the encoder is the only expensive stage.

**Replay.** SAN into board states with python-chess. The rest of PrepMate only
ever stores move strings, so nothing downstream could see a position until this
existed.

**Select.** Keep only the positions where the opponent actually chose something.
That drops the opponent's own half of the plies, the book, the long tail past
move thirty, positions with one legal move, and sole recaptures. A forced retake
tells me nothing about how someone thinks.

**Canonicalise.** Mirror so the side to move always sits at the bottom in white's
colours, then hash. A structure now embeds identically whether it was met as
White or as Black, and transpositions collapse to one key because the hash
ignores move counters.

**Deduplicate and cache.** One embedding per distinct position, stored by
position rather than by player, since the encoder sees a board and nothing else.

**Encode.** Batched forward passes through the frozen trunk.

**Mine.** Spherical k means over the unit vectors, then score each cluster.

## What comes out

**Archetypes.** Recurring position types with the opponent's real score from
each. Ranked by points dropped below their own baseline rather than by
percentage, because prep time is finite and a wide mildly bad structure is worth
more attention than a narrow catastrophic one.

**Predictability.** Mean surprisal in bits of their actual moves under the
population model, plus top one and top three hit rates. High predictability means
prep will land. Low means they improvise and I should prepare positions rather
than lines.

**Retrieval.** Given a position I expect on the board, their most similar
positions and how those went.

## Two statistics I was careful about

**Pseudo replication.** One game can pass through the same structure fifteen
times. Counting those as fifteen observations would inflate every sample size and
make noise look certain, so a game counts at most once per cluster.

**Shrinkage.** `analysis.py` uses a hard cutoff at fifteen games, which treats a
fifteen game bucket and a sixty game bucket as equally trustworthy. Here each
cluster's score is pulled toward the player's own baseline with twelve pseudo
games of prior weight, so small clusters have to earn their deviation. A three
game cluster at zero percent reports around forty percent against a fifty percent
baseline, which is roughly what three games actually justify believing.

## What this costs

Measured on 400 rated games from an active titled player, 36,764 plies.

| stage | count | note |
| --- | --- | --- |
| replay | 36,764 plies | about 1.1s, unavoidable |
| select | 9,405 decisions | 75% dropped |
| deduplicate | 8,769 positions | only 7% |
| encode | 8,769 first time, 0 on a re-scout | the expensive stage |
| mine | about 20ms | a few matrix multiplies |

Selection is the win, not deduplication. I had assumed a repertoire would repeat
itself enough for deduplication to matter, and it does not, because the repeats
live in the opening and the opening is skipped. I also assumed the position cache
would pay off across different players and measured that at 0.35% overlap, which
is nothing. By move eight two players have almost no positions in common.

What the cache does buy is re-scouting. Reopening a dossier or adding a week of
games encodes only what is new, since a position's embedding never changes.

Clustering is brute force on purpose. At a few thousand positions and 128
dimensions the whole thing is a handful of matrix multiplies, so an approximate
nearest neighbour index would add a dependency and save microseconds against an
encoder that costs seconds.

## What I do not trust

The z score on each archetype is a ranking aid and not a p value. The clusters
were chosen by looking at the same games they are then scored on, and there are
a couple of dozen of them, so the selection effect is real. I read anything under
two as noise.

Cluster labels are inferred from the most common openings feeding them plus an
exemplar position. That is a description, not a name, and sometimes a cluster is
just a phase of the game rather than a meaningful structure.

The corpus I trained on is small. Fourteen accounts and 2,652 games gives 163,310
positions, which is enough to demonstrate that the pipeline works end to end but
well short of what the encoder deserves. The right corpus is the Lichess monthly
database dump. A small corpus from few players also risks the trunk learning
those specific players rather than the population.

The whole thing only sees online games. Someone who plays a sharp online
repertoire and a solid one over the board will be misread, and nothing here can
detect that.
