# 2026 app archive

Prepared observations and cached next-event forecasts for **218 matches, 869 available quarters and all 18 clubs**, from 5 March to 26 September 2026. The archive contains 357,223 events and is the default data source for the hosted app and a fresh checkout.

Three quarters are excluded because their recorded event timestamps run backwards: `CD_M20260140001` Q1, `CD_M20260140206` Q2 and `CD_M20260142005` Q3. The app lists only available quarters and reports the coverage of each selected match.

Forecasts come from the frozen tuned decoder-only Transformer, predeclared seed 2026, trained on 2023–2025. No model weights or training are needed to browse this bundle. The manifest preserves the source provenance, model fingerprint and file checksums. The event file uses lossless Zstandard compression; its observations and predictions are unchanged from the prepared local archive.

Data: [torpdata match-event archive](https://github.com/peteowen1/torpdata/releases/tag/chains-data). See the [app guide](../../../docs/app.md) for interpretation and reproduction instructions. The smaller [sample bundle](../opposition-demo/) remains available as an optional demonstration.
