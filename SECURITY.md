# Security posture — MyoMeasure container image

`myomeasure:latest` (linux/amd64). Vulnerability scan: **Docker Scout** (`docker scout cves`, SARIF format), 2026-06-30.

This image is a reproducibility artifact whose scientific/JVM dependency stack is **pinned** (`conda-lock.yml` plus the Bio-Formats `ome:formats-gpl:6.7.0` JAR set). Every finding is triaged into one of three buckets:

- **A — OS/apt layer, fixable** → patched in the Dockerfile via a pinned `apt-get --only-upgrade` of only the affected Ubuntu *noble* packages.
- **B — pinned scientific/JVM stack** (PyTorch, Cellpose, NumPy/SciPy, scyjava/jgo/jpype, the Bio-Formats JARs and their transitive deps) → **not modified**; these pins are load-bearing for reproducibility.
- **C — everything else** (no upstream/apt patch available, or build-only tooling that is unreachable in the offline runtime) → **not modified**.

Runtime reachability throughout is judged against the only things the container does at run time: **offline Cellpose (cpsam) inference on local TIFFs**, **local `.vsi`→TIFF conversion via Bio-Formats**, and the **phantom test** — all with pre-baked weights/JARs and no network.

## Before / after (this hardening pass)

| Severity | Before | After | Δ |
|---|---:|---:|---:|
| Critical | 12 | 12 | +0 |
| High | 68 | 66 | -2 |
| Medium | 87 | 42 | -45 |
| Low | 53 | 18 | -35 |
| **Total** | **220** | **138** | **-82** |

| Ecosystem | Before | After | Δ | Buckets |
|---|---:|---:|---:|---|
| deb | 106 | 24 | -82 | A (fixed) + C (unfixable) |
| maven | 92 | 92 | +0 | B (Bio-Formats JARs) |
| pypi | 22 | 22 | +0 | C (base-env tooling) + B (runtime deps) |

All **82** fixable OS CVEs were cleared (including both deb *High* findings); the pinned scientific/JVM stack (B) and the unfixable/unreachable set (C) are unchanged by design. The phantom-test build gate still passes (11/11), and the base image remains digest-pinned.

## Bucket A — OS packages patched (fixed)

A dedicated early Dockerfile layer runs `apt-get update && apt-get install -y --only-upgrade --no-install-recommends <pinned set>` over **only** the CVE-affected *noble* packages, each pinned to its noble-security version (30 binaries across 18 source packages):

`curl` · `dpkg` · `expat` · `glibc` · `gnupg2` · `gnutls28` · `libcap2` · `libgcrypt20` · `libssh` · `libtasn1-6` · `nghttp2` · `openssl` · `perl` · `sed` · `systemd` · `tar` · `util-linux` · `xz-utils`

Result: deb findings 106 → 24 (**−82**); deb High 2 → 0, Medium 59 → 14, Low 45 → 10. The version pins track the noble-security pocket at scan time; refresh them when re-scanning (a superseded pin fails the build loudly rather than silently drifting).

## Bucket B — pinned scientific/JVM stack (documented, not modified)

Bio-Formats JARs resolved for `ome:formats-gpl:6.7.0` and pre-baked in `/root/.jgo`. Upgrading any of them would change the pinned Bio-Formats stack and break reproducibility, so they are retained and documented. Every Critical/High:

| Package | Sev | #CVE | Runtime reachability | One-line rationale |
|---|---|---:|---|---|
| `com.fasterxml.jackson.core/jackson-databind` | Critical | 51 | loaded, dormant | On the jgo/Bio-Formats classpath but no ObjectMapper/JSON deserialization runs on any offline path; VSI reads parse binary pixels + OME-XML (SAX/DOM), not JSON. |
| `org.json/json` | High | 4 | loaded, dormant | On the Bio-Formats JVM classpath but local VSI reads use binary CellSens decode + OME-XML (xerces); no JSON parser is invoked. |
| `xerces/xercesImpl` | High | 2 | exercised | EXERCISED: local .vsi read parses OME-XML via Bio-Formats and xercesImpl is the bundled JAXP provider, so its parse-time code runs offline on the user's own file. XXE/DoS require attacker-controlled XML, which a self-supplied microscopy file is not. |
| `ch.qos.logback/logback-classic` | High | 2 | loaded, dormant | Loaded as the SLF4J backend and emits benign log lines, but the CVE sinks (Janino/config parsing, SocketNode receiver deserialization) need attacker-controlled logback config or a network receiver, neither of which exists. |
| `com.mchange/c3p0` | High | 2 | loaded, dormant | JDBC connection pool; no container op opens a DB and OME-XML uses xerces not c3p0, so its XXE/DoS config code never runs. |
| `com.drewnoakes/metadata-extractor` | High | 1 | loaded, dormant | In the Bio-Formats fat-jar classpath but CellSensReader parses ETS tiles + OME-XML via Xerces and never invokes com.drew EXIF parsers. |
| `com.squareup.okhttp3/okhttp` | High | 1 | unreachable | HTTP client that only backs remote/OMERO/URL readers never used for a local file; classes are never even loaded on the three offline paths. |
| `xalan/xalan` | High | 1 | loaded, dormant | Can ride the classpath, but a local .vsi builds OME-XML programmatically and parses via DOM/SAX; no untrusted-stylesheet XSLT (CVE-2022-34169 trigger) runs. |
| `com.fasterxml.jackson.core/jackson-core` | High | 1 | loaded, dormant | On the JVM classpath but local VSI reads parse binary pixels + OME-XML (xerces) and use slf4j logging, never jackson; no JSON parse path executes. |
| `io.minio/minio` | High | 1 | unreachable | s3://-only HTTP client; offline image has no S3 endpoint and local file reads never reference minio, so the class is never loaded. |
| `org.xerial/sqlite-jdbc` | High | 1 | unreachable | JDBC driver only class-loaded when a SQLite-backed reader opens a DB via DriverManager; VSI/OME-XML/logging never touch java.sql, so the jar stays unloaded. |
| `com.mchange/mchange-commons-java` | High | 1 | loaded, dormant | c3p0/JDBC-only utility; local .vsi read uses xerces OME-XML + slf4j (not MLog) and no DB, so its deserialization CVE code is never invoked. |

## Bucket C — no patch available / unreachable tooling (documented, not modified)

**pypi (base build-env, Python 3.12):** these back `conda-lock`/`mamba`/`gitpython`/`matplotlib` in the *base* conda env — a different interpreter than the `myomeasure` runtime env (3.11) — so they are dormant or never imported at run time. Every Critical/High:

| Package | Sev | #CVE | Runtime reachability | One-line rationale |
|---|---|---:|---|---|
| `urllib3` | High | 4 | loaded, dormant | Transitively resident (fsspec/requests) when bioio is imported, but local .vsi reads issue no HTTP, so its request/redirect/TLS CVE code never executes. |
| `cryptography` | High | 2 | unreachable | Importing the full runtime stack (bioio+scyjava+jpype+cellpose) loads zero pyca-cryptography modules; Bio-Formats crypto is Java-side; inference/test are pure torch/numpy. |
| `dulwich` | High | 2 | unreachable | Pure-Python Git plumbing confined to base-env build tooling (conda-lock/gitpython); runtime paths import no git. |
| `wheel` | High | 1 | unreachable | PEP 427 build/install packaging tooling; no runtime path (inference, VSI conversion, phantom test) imports it. |
| `brotli` | High | 1 | unreachable | Transitive dep of matplotlib via fonttools (WOFF2 font codec); matplotlib Agg overlays use FreeType TTF and never import fontTools.ttLib.woff2, so brotli is never loaded. |

**deb (residual, unfixable):** after the bucket-A upgrade, **24** deb findings remain — all *Medium*/*Low*, **zero High/Critical**. They have no fixed version in the noble archive: `git`, `wget`, `zlib`, `shadow` have no candidate at all; `curl`/`glibc`/`expat`/`dpkg`/`tar`/`util-linux`/`systemd`/`libgcrypt20` are already at their latest patched version with residual un-patched CVEs. They clear automatically once Ubuntu publishes fixes and the pins are refreshed.

Residual deb (source → findings): `curl` 9 · `dpkg` 1 · `expat` 1 · `git` 1 · `glibc` 5 · `libgcrypt20` 1 · `shadow` 1 · `systemd` 1 · `tar` 1 · `util-linux` 1 · `wget` 1 · `zlib` 1

## Appendix — every Critical/High in buckets B and C

| CVE / advisory | Package | Sev | Bucket | Reachability |
|---|---|---|---|---|
| CVE-2019-14379 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2019-14540 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2019-16335 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2019-16942 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2019-16943 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2019-17267 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2019-17531 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2019-20330 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2020-8840 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2020-9546 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2020-9547 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2020-9548 | `com.fasterxml.jackson.core/jackson-databind` | Critical | B | loaded, dormant |
| CVE-2023-6378 | `ch.qos.logback/logback-classic` | High | B | loaded, dormant |
| CVE-2023-6378 | `ch.qos.logback/logback-classic` | High | B | loaded, dormant |
| CVE-2022-24614 | `com.drewnoakes/metadata-extractor` | High | B | loaded, dormant |
| CVE-2025-52999 | `com.fasterxml.jackson.core/jackson-core` | High | B | loaded, dormant |
| CVE-2019-12086 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2019-14439 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2019-14892 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2019-14893 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-10650 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-10672 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-10673 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-10968 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-10969 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-11111 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-11112 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-11113 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-11619 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-11620 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-14060 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-14061 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-14062 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-14195 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-24616 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-24750 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-25649 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-35490 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-35491 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-35728 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36179 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36180 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36181 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36182 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36183 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36184 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36185 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36186 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36187 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36188 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36189 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2020-36518 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2021-20190 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2022-42003 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2022-42004 | `com.fasterxml.jackson.core/jackson-databind` | High | B | loaded, dormant |
| CVE-2019-5427 | `com.mchange/c3p0` | High | B | loaded, dormant |
| CVE-2026-27830 | `com.mchange/c3p0` | High | B | loaded, dormant |
| CVE-2026-27727 | `com.mchange/mchange-commons-java` | High | B | loaded, dormant |
| CVE-2021-0341 | `com.squareup.okhttp3/okhttp` | High | B | unreachable |
| CVE-2025-59952 | `io.minio/minio` | High | B | unreachable |
| CVE-2022-45688 | `org.json/json` | High | B | loaded, dormant |
| CVE-2022-45689 | `org.json/json` | High | B | loaded, dormant |
| CVE-2022-45690 | `org.json/json` | High | B | loaded, dormant |
| CVE-2023-5072 | `org.json/json` | High | B | loaded, dormant |
| CVE-2023-32697 | `org.xerial/sqlite-jdbc` | High | B | unreachable |
| CVE-2022-34169 | `xalan/xalan` | High | B | loaded, dormant |
| CVE-2012-0881 | `xerces/xercesImpl` | High | B | exercised |
| CVE-2013-4002 | `xerces/xercesImpl` | High | B | exercised |
| CVE-2025-6176 | `brotli` | High | C | unreachable |
| CVE-2026-26007 | `cryptography` | High | C | unreachable |
| GHSA-537c-gmf6-5ccf | `cryptography` | High | C | unreachable |
| CVE-2026-42305 | `dulwich` | High | C | unreachable |
| CVE-2026-42563 | `dulwich` | High | C | unreachable |
| CVE-2025-66418 | `urllib3` | High | C | loaded, dormant |
| CVE-2025-66471 | `urllib3` | High | C | loaded, dormant |
| CVE-2026-21441 | `urllib3` | High | C | loaded, dormant |
| CVE-2026-44431 | `urllib3` | High | C | loaded, dormant |
| CVE-2026-24049 | `wheel` | High | C | unreachable |

*78 Critical/High findings across 17 packages (12 in B, 5 in C). Full machine-readable reports: `scout.json` (fixable subset) and `scout_all.json` (all findings).*

## Reproduce / refresh

    docker scout cves myomeasure:latest --platform linux/amd64 --format sarif --only-fixed --output scout.json
    docker scout cves myomeasure:latest --platform linux/amd64 --format sarif --output scout_all.json
    # refresh bucket-A pins: candidate noble-security versions for the affected packages
    docker run --rm --platform linux/amd64 <base-image-digest> bash -c 'apt-get update && apt-get -s upgrade'
