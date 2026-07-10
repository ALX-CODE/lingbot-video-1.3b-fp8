# Security policy

## Supported versions

Security fixes are applied to the latest published release and the default
branch. Older releases may receive a fix when the change can be backported
safely.

## Reporting a vulnerability

Do not disclose a suspected vulnerability, credential, private path, or
exploit in a public issue. Use GitHub's **Report a vulnerability** flow under
the repository Security tab:

https://github.com/ALX-CODE/lingbot-video-1.3b-fp8/security/advisories/new

If private vulnerability reporting is unavailable, open a minimal issue asking
the maintainer for a private contact channel without including technical
details.

Include the affected version or commit, impact, reproduction conditions, and a
suggested mitigation when available. Please allow reasonable time for triage
and coordinated disclosure.

## Trust boundaries

- Model weights and configuration files are data inputs. Prefer Safetensors,
  verify published SHA-256 digests, and obtain assets only from the documented
  upstream and release locations.
- Custom nodes execute Python with the same privileges as ComfyUI. Review
  dependency and custom-node sources before installation.
- Never attach private prompts, source images, generated media, credentials, or
  complete machine paths to public reports without sanitizing them.
- This project does not silently replace PyTorch. Treat CUDA/PyTorch upgrades
  as separate system-administration changes.
