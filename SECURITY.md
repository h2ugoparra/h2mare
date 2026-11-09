# Security Policy

## Supported versions

h2mare is pre-1.0 and maintained by one person. Security fixes go into the
latest released minor version only; upgrade to it to receive them.

| Version | Supported |
| ------- | --------- |
| 0.10.x  | ✅        |
| < 0.10  | ❌        |

## Reporting a vulnerability

**Please do not open a public issue, discussion or pull request for a
security problem.**

Report it privately through GitHub instead:
[**Report a vulnerability**](https://github.com/h2ugoparra/h2mare/security/advisories/new)
(Security tab → *Report a vulnerability*).

Please include:

- the h2mare version (`pip show h2mare` or `uv pip show h2mare`) and Python version;
- what an attacker can do, and what they need to control to do it
  (a config file, an input file, a network response, …);
- steps or a minimal file/config that reproduces it.

## What to expect

This is a best-effort, single-maintainer project, so these are targets rather
than guarantees:

- acknowledgement within **7 days**;
- an initial assessment within **14 days**;
- for a confirmed issue, a fix released as a patch version, followed by a
  GitHub Security Advisory crediting you (unless you prefer not to be named).

Please give us a reasonable chance to release a fix before disclosing publicly.

## Scope

In scope — problems in h2mare's own code, for example:

- leaking credentials from `.env` (AVISO FTP, CDS, Copernicus Marine) into
  logs, error messages, written files or exceptions;
- code execution or writes outside the configured store when processing a
  crafted `config.yaml`, NetCDF/Zarr/Parquet file, shapefile or CSV;
- path handling that lets a variable key, `local_folder` or `store_root`
  escape its intended directory;
- weaknesses in the CI or release workflows that could compromise the
  published PyPI package.

Out of scope:

- vulnerabilities in dependencies (xarray, zarr, dask, copernicusmarine,
  cdsapi, …) — report those upstream; we will bump the dependency once a fix
  is released;
- the data providers' services themselves (CMEMS, AVISO, CDS);
- issues that require an attacker who already controls your machine, your
  `.env`, or your Python environment.
