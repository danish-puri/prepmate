"""Deep pattern detection over an opponent's recent games.

Entry point is pipeline.build() then pipeline.report(). encoder.load() picks the
pretrained model when a checkpoint exists and a handcrafted fallback otherwise.
"""
