# syntax=docker/dockerfile:1
#
# MyoMeasure inference + VSI-conversion container.
# Lets others run run_inference.py (Cellpose, cpsam) and convert_vsi_to_tiff.py
# (Olympus .vsi -> calibrated TIFF) from a pinned, reproducible environment.
#
# The environment is installed FROM conda-lock.yml (generated cross-platform:
# osx-arm64 + linux-64). This image is linux-64, so build for that platform:
#
#   docker build --platform linux/amd64 -t myomeasure:latest .
#
# A successful build means the synthetic-phantom pytest gate (step 7) passed.

# --- Base pinned BY DIGEST (not tag) ---------------------------------------
# condaforge/miniforge3:25.3.1-0 multi-arch index digest. Build selects the
# linux/amd64 child (sha256:4dea90c1ceae13632fd1da60ac64554e90c9cfb28f4aa937c9339f5226738c2f)
# via --platform linux/amd64. Re-resolve with:
#   docker buildx imagetools inspect condaforge/miniforge3:25.3.1-0 --format '{{.Manifest.Digest}}'
FROM condaforge/miniforge3@sha256:1e65803d646bf8503728001dddb5756f3e8235e0883c19b5a5ad1da85cff0905

# 0. OS security hardening (bucket A). Patch the FIXABLE Ubuntu 'noble' CVEs by
#    upgrading ONLY the CVE-affected apt packages (from `docker scout cves`,
#    package type = deb, fixable) to their pinned noble-security versions. The
#    base image stays digest-pinned above; this layer applies the OS patches on
#    top of it. Version pins track the noble-security pocket at scan time and
#    should be refreshed when re-scanning (see SECURITY.md). All other findings
#    are in the pinned scientific/JVM stack (bucket B) or are unfixable/
#    unreachable (bucket C); those are documented in SECURITY.md, not modified.
RUN apt-get update && \
    apt-get install -y --only-upgrade --no-install-recommends \
    libc6=2.39-0ubuntu8.7 libc-bin=2.39-0ubuntu8.7 \
    openssl=3.0.13-0ubuntu3.11 libssl3t64=3.0.13-0ubuntu3.11 \
    libgnutls30t64=3.8.3-1.1ubuntu3.6 libgcrypt20=1.10.3-2ubuntu0.1 \
    libtasn1-6=4.19.0-3ubuntu0.24.04.2 libssh-4=0.10.6-2ubuntu0.4 \
    libnghttp2-14=1.59.0-1ubuntu0.3 libcurl3t64-gnutls=8.5.0-2ubuntu10.10 \
    libexpat1=2.6.1-2ubuntu0.4 liblzma5=5.6.1+really5.4.5-1ubuntu0.3 \
    libcap2=1:2.66-5ubuntu2.4 gpgv=2.4.4-2ubuntu17.4 \
    libsystemd0=255.4-1ubuntu8.16 libudev1=255.4-1ubuntu8.16 \
    dpkg=1.22.6ubuntu6.6 tar=1.35+dfsg-3ubuntu0.1 sed=4.9-2ubuntu0.24.04.1 \
    perl=5.38.2-3.2ubuntu0.3 perl-base=5.38.2-3.2ubuntu0.3 \
    perl-modules-5.38=5.38.2-3.2ubuntu0.3 libperl5.38t64=5.38.2-3.2ubuntu0.3 \
    util-linux=2.39.3-9ubuntu6.5 bsdutils=1:2.39.3-9ubuntu6.5 \
    mount=2.39.3-9ubuntu6.5 libmount1=2.39.3-9ubuntu6.5 \
    libblkid1=2.39.3-9ubuntu6.5 libsmartcols1=2.39.3-9ubuntu6.5 \
    libuuid1=2.39.3-9ubuntu6.5 && \
    rm -rf /var/lib/apt/lists/*

# 1. Put conda-lock in the base env (conda-forge only, matching the dev setup).
RUN mamba install -n base -c conda-forge -y "conda-lock>=2.5" && mamba clean -afy

WORKDIR /app

# 2. Create the locked env from the cross-platform lock. `conda-lock install`
#    installs BOTH the conda solve AND the pip section (bioio-bioformats) in one
#    step; `conda create --file` of a rendered lock would drop the pip portion.
COPY conda-lock.yml ./
RUN conda-lock install --mamba --name myomeasure conda-lock.yml && mamba clean -afy

# 3. Run every subsequent build step inside the activated env so the conda
#    openjdk activation sets JAVA_HOME for scyjava/jpype (Bio-Formats JVM).
SHELL ["conda", "run", "--no-capture-output", "-n", "myomeasure", "/bin/bash", "-c"]
ENV JAVA_HOME=/opt/conda/envs/myomeasure

# 4. Project code (build context trimmed by .dockerignore).
COPY . /app

# 5. Pre-bake the Bio-Formats Java library (ome:formats-gpl:6.7.0, pinned by
#    bioio-bioformats==1.3.2) into ~/.jgo + ~/.m2 + ~/.cache/cjdk so VSI
#    conversion runs OFFLINE at runtime. jgo's resolver only searches Maven
#    Central and scijava.public, so we add the OME artifactory repo, which hosts
#    the transitive woolz:JWlz artifact. Needs network at BUILD time only.
RUN python -c "import scyjava; scyjava.config.add_repositories({'ome': 'https://artifacts.openmicroscopy.org/artifactory/maven'}); scyjava.config.endpoints.append('ome:formats-gpl:6.7.0'); scyjava.start_jvm(); import bioio_bioformats; print('Bio-Formats JVM + jar cached OK')"

# 6. Pre-bake the Cellpose cpsam model weights into ~/.cellpose so inference runs
#    OFFLINE out of the box (matches run_inference.py --model default 'cpsam').
RUN python -c "from cellpose.models import CellposeModel; CellposeModel(gpu=False, pretrained_model='cpsam'); print('cpsam weights cached OK')"

# 7. BUILD GATE: the synthetic-phantom pytest. A non-zero exit FAILS the build.
RUN pytest tests/ -q

# 8. Entry point: run either script, e.g.
#      docker run --rm myomeasure:latest run_inference.py --help
#      docker run --rm -v "$PWD/data:/app/data" myomeasure:latest convert_vsi_to_tiff.py data/A1
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myomeasure", "python"]
CMD ["run_inference.py", "--help"]
