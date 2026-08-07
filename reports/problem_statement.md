# Fairness & Bias Audit — Adult Income Dataset

## Decision being modeled
Predicting whether an individual's income exceeds $50K/year, based on census
features (education, occupation, hours worked, etc.). This mirrors real-world
use cases like loan pre-screening or hiring-salary-band prediction.

## Who is affected
Predictions are evaluated for disparity across `sex` (Male/Female) and
`race`. A biased model here could translate to real discriminatory impact
if deployed in lending or hiring contexts.

## Why this matters
Under frameworks like the EU AI Act and US EEOC guidance on algorithmic
hiring, models used in employment/credit decisions must be auditable for
disparate impact. This project builds that audit pipeline end-to-end.