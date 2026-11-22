"""Model clients: the model channels, HTTP calls with retries, and tool-calling
conversations in each endpoint's own request format.

A model channel is an endpoint with its own model names and request format;
``ruikit/configs/models.yaml`` lists them:

* ``openai`` (the default) -- OpenAI's API, or any OpenAI-compatible endpoint
  (``OPENAI_BASE_URL``), with the key ``OPENAI_API_KEY``.
* ``anthropic`` -- Anthropic's OpenAI-compatible endpoint, with ``ANTHROPIC_API_KEY``.
* ``local`` -- a model served on this machine by vLLM, OpenAI-compatible.  Its
  thinking is off unless ``RUIKIT_LOCAL_THINKING=1`` (with it on, a reply took
  7-9 minutes); what it thinks comes back apart from its answer and is kept in
  the transcript only.

A model is named ``<channel>:<name>`` (``openai:<model>``) or by a bare name,
which goes to the default channel.  Endpoints and credentials live
in the secrets file (``.secrets/models.env``, gitignored; ``load_settings``); a
variable set in the shell wins over it, and an endpoint set in either wins over
the one in the YAML.

Requests go through ``urllib``, with no vendor SDK.  A conversation keeps its
history in the shape of the endpoint it talks to (chat completions or
responses), so a tool loop only says what happened: the model's calls, their
results, the pictures.
"""
