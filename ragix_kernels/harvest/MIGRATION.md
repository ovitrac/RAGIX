# MIGRATION — the harvest family, from the lab's tree into RAGIX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Source: the lab's repository at `1f440cba74c9807745964ee5c0ae2b19da1dc4ad`. Each module moved as it is; the only edits are the
imports (the lab's `sys.path` arrangements became package-relative), the author line, and the scrub:
the consultation's buyer and reference, the internal library's name, home paths and private
addresses became neutral words or environment variables (see `README.md`). The three runner
sources were merged into one module, `runner.py`, with renamed entry points and the host as an
argument; that is the one exception to « as it is ».

## Modules

| source | sha256 of the source | target (`ragix_kernels/harvest/…`) |
|---|---|---|
| `src/tender/dates_fr.py` | `e6756ff47ea67c2c0d5115f2ff3f4887f85176234afdbe0576fabaf7135e1ba7` | `fr/dates.py` |
| `src/tender/grammars_fr.py` | `72e6551701b604fabbdf749ff8ef1bc662103ff64e7aec55359d50fabcfb50ae` | `fr/grammars.py` |
| `src/tender/cut.py` | `6c6e990e23f4d559a5dcaa6a429741932ded4559142d80530383385cbed481aa` | `fr/cut.py` |
| `src/tender/harvest.py` | `94d9db7fe9d7d281823d8dc4763ddb80ab59bb5c3b739bf7e5e30de26ba1b8a7` | `form.py` |
| `demoE2E/16_harvest_bakeoff/harvest_runner.py` | `376a3bff75dc078dfe6a61deb08967f163b9f76a4d8212ffd6deef0b611f65a1` | `runner.py (bake-off: `run_bakeoff`)` |
| `demoE2E/19_node_harvest/harvest_nodes.py` | `0a972d1c14f7d0da0b743705339dad7eb6151ef63533e1b0830f79a4cf8e4216` | `runner.py (node roll-ups: `run_nodes`)` |
| `demoE2E/25_form_8b/form_pass2.py` | `5dd83f972d40b6fe7b7dc3a3883b5cf4179856dd4b5b4ec740bae02595df428c` | `runner.py (pass 2 at window grain: `run_window`)` |
| `src/tender/derived.py` | `0eec608d184b3d786decdc7a45da8a8ab97b555bd3a07bdca8d8f94422cdf499` | `derived.py` |
| `demoE2E/verify/pass1/pass1.py` | `10d4236c1a9c0f5b2efa3446574b22e2a2c473d7893f1f4210538074f3660b2f` | `pass1.py` |
| `demoE2E/verify/pass1/core.py` | `b735bf795f690917f932738ef7db4b2b58cb9fdfb509ad6aa2f9a74b09506e3a` | `core.py` |
| `demoE2E/03_analyze/analyze.py` | `d3779bdb0a485d50539872b6e8a2a2218b344df91fb0ffcaa4db711f23222004` | `pieces.py (`load_pieces` only)` |
| `demoE2E/verify/family/families.py` | `f5f5c59b049473df7c1dca4af86cf5a2d987a6b209235c7be937ad9aac41bd82` | `families.py` |
| `demoE2E/26_embed/embed_abstracts.py` | `afb63bd00d31075324aa7302e1b84d479343d4ce3030ba6a3eb55be347b2bd7a` | `embed_abstracts.py` |
| `demoE2E/verify/pass1/map_test.py` | `5adafab980a33366b5054ee407ae70afc271e834d84ff32f0a0e52544d8d2d57` | `map_test.py` |
| `demoE2E/28_descent/descent.py` | `e7986921e8fb2cdc16531a3fe6dad0d4311fb5d86956b055f76383170c1e1e3c` | `descent.py` |
| `demoE2E/27_graph/build_edges.py` | `6f7de844d28a00a965b8af3e782d73745adad4b32aea16e475be9febecf70386` | `build_edges.py` |
| `demoE2E/29_ask/ask.py` | `8bff3d671f4e12b4e6ab14bfe5d69609242e0d7ee37364602859a66366b8c095` | `ask.py` |
| `demoE2E/25_form_8b/rescore_repeats.py` | `024254088bd2d0c269d2529fb5aa45b0c94027f7f1fed61696a15755ae481cd6` | `rescore_repeats.py` |
| `demoE2E/verify/pass1/rescore.py` | `8709378d3b9046aea4da4e776921ac82a0567beb101e4ecf8437f1381354a9b1` | `rescore.py` |
| `demoE2E/verify/brief/check_brief.py` | `1eb88ec69d9957ab8746a4a65f4c6bd54e304635f39894e04f0c9427d22c3f54` | `check_brief.py` |
| `demoE2E/verify/brief/scan_cut.py` | `5094bb6dcff358b2952cf221e1006ca46acdb92452c37920910431b2665748fd` | `scan_cut.py` |
| `demoE2E/verify/family/family.py` | `a6a2f23d303ecc6ae714e286b7c9861f5f954b1c55c95e0ec575c70c0f8c8d28` | `registers/family.py` |
| `demoE2E/verify/family/commitments.py` | `721bc4a648c2c89db14578d73c25e5c4d17383b32a802015711af5d5f0dc9c59` | `registers/commitments.py` |
| `demoE2E/verify/family/clauses.py` | `3d0d33319922030f6d29ddfcd8aae3bbde02c45ff06c530595c544cb7bdf71b3` | `registers/clauses.py` |
| `demoE2E/verify/family/template.py` | `b0e0bf7264f0e19b59f634e6f3383b956e15bf9bf34355a3d138ecc8a5e09f4a` | `registers/template.py` |
| `demoE2E/verify/family/traps.py` | `e1c277c3bcfa77c174b23973237f1f3a9684adb40c7f040c813512cf0ece1601` | `registers/traps.py` |
| `demoE2E/verify/family/render_fr.py` | `11b4e37fdd933f39edae8e889f6d07e5b01b2c5e45eaca0b9f999b0950bf55ea` | `registers/render_fr.py` |
| `demoE2E/00_provenance/provenance.py` | `4a7e4c8858026b760278f3a1d5617ca9b333886dc15f145ea0d47dd1919d74bb` | `provenance.py` |
| `demoE2E/journal.py` | `168a2b3e475d7e128613b278b82e72a14336412c30a4fc9a2aa0304429a75502` | `../shared/journal.py` |
| `demoE2E/gold/read.py` | `b6c42ec7f7b6d41e549c04e1428c70890e9882e5622389a0d28a56a0eaa32f09` | `gold/read.py` |
| `demoE2E/gold/agreement.py` | `dc5e8b0f3b78f4a90ae84d7500f277b91bcce322e35444b7a568b6356b8ba6d8` | `gold/agreement.py` |
| `demoE2E/gold/select_pass2.py` | `e0f2d9e5a08be6269b700f38c475745fa16c52065c8a83995a5fddfd5e6e681a` | `gold/select_pass2.py` |
| `demoE2E/gold/coord_check.py` | `662ea075f0609023df7a5b8caf3ef4914b1727c9e1c60188918416e05877c372` | `gold/coord_check.py` |

Bench files, which were in no repository (sha256 of the copies migrated):

| source | sha256 | target |
|---|---|---|
| `granite42/Modelfile.8b-chat` | `b838df80134eba710be8333c05475b4ef30ff073e934d38af674a13faef7dac5` | `models/granite42/Modelfile.8b-chat` |
| `granite42/Modelfile.chatml` | `e19bd533126767d86df87e2fc1a237dd18fc70304f1f10efab94a0069459d976` | `models/granite42/Modelfile.30b-chat` |
| `granite42/granite42_chatml.tmpl` | `7704978f5840f938c0114c431422f6819309f0c40d5c5485e8d55b54020d82c4` | `models/granite42/granite42_chatml.tmpl` |
| `granite42/manifest.py` | `396f7719aab73b387d9169eb7af798e88552552b336bdb637eea3a3b91efa6d8` | `models/manifest.py` |
| `ps_watch.sh` | `a3c88277b9fd6a209ef94035da49224073b89ed1ce6c3818292cde2bae1633cf` | `models/ps_watch.sh` |
| `conc_watch.py` | `f0b41f46c5b84994c1456b79aa7e5b89533a5be99b41165a5556f1503679894a` | `models/conc_watch.py` |
| `harvest.sh` | `a96e7696e2f0c7b98bf55bf15e186d5699a1767653dfa44a37462c8f582a6f45` | `models/ollama_journal.sh` |

## Tests ported

| source | sha256 of the source | target (`tests/harvest/…`) |
|---|---|---|
| `tests/test_grammars_fr.py` | `fa2e56784ba5b59b9558a2f9212ef7782ec181accc6827b2acc055ea71952c31` | `test_fr_grammars.py` |
| `tests/test_grammars_13.py` | `dfffe606bff409f70e1284782c0bb0f267d8d0cd3f0cc7e683751af733b11f7e` | `test_fr_grammars.py` |
| `tests/test_grammars_14.py` | `d7ca287a34b3f7e4a3200f7e803461c48bc8a2d0831b83cccd00358718a8feca` | `test_fr_grammars.py` |
| `tests/test_grammars_15.py` | `1757680b7d70ac219c6942668d70ffac34226d500879ffe640bb4e9c172bac68` | `test_fr_grammars.py (the store scan stayed: it measures the lab's store)` |
| `tests/test_grammars_hours.py` | `6827e63d31b616b5a33ab2a54397492c6bbed632457342f55a9bec11a0418e00` | `test_fr_grammars.py` |
| `tests/test_dates_fr.py` | `f6a9ed39bd5e036945d78ae919e70ac9033d87c37eb093ca0a48e54efce3b0f9` | `test_fr_dates.py (the grammar half; the deadline-slice half is the tender domain's)` |
| `tests/test_joined_values.py` | `5fc72d2fa977f7054b15e9f39e4aff4f8c53000899aaab8db635a374799cd525` | `test_fr_cut.py (the `joined_runs` test; the rest drives build_brief, which stayed)` |
| `tests/test_harvest.py` | `30c34cbe2b865010965e3eb8e0feb8d23f78b4e5c9cf7b06c1b362aaebe9fe85` | `test_form.py` |
| `tests/test_form_markers.py` | `10b19ed324226e6ad7644a32bd04228fdc4af846501af8673f10dc331164753d` | `test_form_markers.py` |
| `tests/test_form_grain.py` | `3a799a6bb8e054aec50bc2b05d6b46a2eb053e88d6446fe75550eff651c99e80` | `test_form_grain.py` |
| `tests/test_form_hollow.py` | `7e850f42227825818bebbf7e6faee511e378e0e3a29e82987020364733a4901b` | `test_form_hollow.py` |
| `tests/test_harvest_substitute.py` | `e012822c0033f58bd81c405172d5edd02199dffb63749f1cb38ac19eb9c0c347` | `test_form_substitute.py (the recorded-answers test stayed)` |
| `tests/test_harvest_runner.py` | `cb9f197605f25cc94d4ef135e25d6525af9fa9003424ac33b60fcc94092fe4e9` | `test_runner_bakeoff.py` |
| `tests/test_form_pass2.py` | `1f0e4e16ec0685476191936b360301566792dcd6c3f73a3bf52db3987e30f793` | `test_runner_window.py (the recorded-answers test stayed)` |
| `tests/test_derived_embeddings.py` | `33945e2001f208e3a96c907b4b24063c56808c0050bfd08963b9abca37eac481` | `test_derived_embeddings.py` |
| `tests/test_pass1_ladder.py` | `d831f436d2760df18f22e66a3a10ebc20770ce17893fc512c5a0fe39d0759788` | `test_pass1_ladder.py` |
| `tests/test_pass1_truncation.py` | `d94e25858621adfb8959ce79c52f35fe14d88e7915c62110e5ab395ee51e93d8` | `test_pass1_truncation.py` |
| `tests/test_core_records.py` | `77526b4c9802d8590cef785881ca0abdcd56de4cd726627f08fd90547235bd3f` | `test_core_records.py` |
| `tests/test_core_workers.py` | `40b5c5ca461b66435458d3b3dce9e7861452f172348fbf5d6462ed62589a85c8` | `test_core_workers.py` |
| `tests/test_family_stage.py` | `4a0d28e242f17b9d59edd6d90ae834b92fd4e6e7b2ff7391e5d8753fe5d965e8` | `test_family_stage.py` |
| `tests/test_embed_abstracts.py` | `93a975f06c2f174339d8929cdbcd16a66c5e2b42145827677c7305fcc84d80cb` | `test_embed_abstracts.py` |
| `tests/test_descent.py` | `8bc6718ee8e05b1d6731520a32c1a68a7f8efd418ed69075df8897c72e4dc203` | `test_descent.py (the embed_queries and graph_queries tests stayed with those modules)` |
| `tests/test_graph_edges.py` | `4ecd83b737c83033ca958c3636c275f4e7a2ac6e615b215237245e6ffca0bb1c` | `test_graph_edges.py` |

Added with the move: `synthetic.py` (synthetic stores built through the store's own API),
`test_fr_cut.py` (the cut rule), `test_runner_nodes.py` and `test_runner_window_job.py` (the runner's
jobs end to end against a fake server), `test_imports.py` (every module imports from the package).

## What stayed in the lab, and why

- the tender domain library (`src/tender/` records, claims, contract, …): moved separately, to `ragix_kernels/tender/domain/`;
- `demoE2E/20_brief/build_brief.py` and `28_node_cards/build_cards.py`: they compose one consultation's brief and the player's cards;
- `demoE2E/28_descent/embed_queries.py`, `graph_queries.py`, `16_harvest_bakeoff/score_harvest.py`, `verify/family/posthoc.py`, `verify/family/family.py`'s neighbours `pyramid_svg.py`, `verify/pass1/check_rules.py`, `smoke.py`, `manager_questions.py`: measurements and one-off analyses of the lab's runs;
- the player, the runbook launchers, the registry, and everything under the in-tree substrate library, which is not public;
- the bench's grafted `Modelfile` (the 4.0/4.1 template on 4.2, which the bench itself advised against), `Modelfile.noparams` (a machine's blob path), `docx_probe.py` (a one-off diagnostic reading a document outside the store).

## Needs the owner's validation

- the reading of the consultation's reference from `HARVEST_CONSULTATION` (and the other environment variables) where the code matched a file name or a header: the code names no consultation, and the owner sets the variable to run it on one;
- the renamed manifest keys in `provenance` (`record_zip`, `manifest_record`, `record_files`), forced by the repository guard's hashed tokens;
- kernel wrappers, MCP tools and the manual (`docs/KOAS_HARVEST.md`), deferred to follow-ups by the owner's ruling.
