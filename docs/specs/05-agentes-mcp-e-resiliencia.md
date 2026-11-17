# Spec 05 — Agentes, MCP e resiliência

## Modos

- `EVENTOPS_MODE` seleciona o perfil do produto e permanece `offline` no demo.
- `EVENTOPS_AGENT_MODE=offline` dá respostas determinísticas e não chama APIs;
  o coordenador offline continua executando pelo `Runner` e grafo de agentes ADK.
- `local` usa Ollama através de um adaptador `BaseLlm` do ADK; `live` usa
  Gemini/Vertex pelo ADK. Credenciais são passadas a clientes por requisição.
- A interface pode escolher arquitetura `single` ou `multi`. No multi, um
  `Agent` coordenador ADK transfere para especialistas de agenda, FAQ ou fluxo
  combinado. `transfer_to_agent` e chamadas de ferramenta aparecem no trace.
  Agenda usa MCP stdio quando configurado; FAQ consulta documentos verificados.

## Limites de segurança

- Ferramentas de agente são somente de leitura. O agente pode propor reserva,
  mas só a interface/API de preview e confirmação pode gravá-la.
- Saídas são validadas contra IDs, fontes e schema conhecidos. Fonte desconhecida,
  sessão conflitante ou campos extras não são aceitos como resultado válido.
- Limite de ferramentas/delegações é por requisição. Harness de falha é interno
  aos testes e não pode ser ativado por mensagem pública.
- O MCP usa o mesmo banco/catalogo configurado, valida ferramentas e payloads,
  fecha o processo/transporte após uso e trata cancelamento/falha.

## Resiliência

- Deadline total limita a duração da chamada; retry só ocorre em falhas
  transitórias elegíveis e dentro do deadline.
- Circuit breaker do provider e proteção do transporte MCP impedem repetição
  contínua em falhas. Falhas não são reportadas como sucesso.
- O adaptador Ollama converte chamadas de ferramenta entre a API de chat local
  e o contrato `BaseLlm` do ADK.

## Código e validação

- Providers e agente: `app/agent.py`, `app/config.py`.
- Coordenador: `app/agents/coordinator.py`.
- MCP: `app/mcp_client.py`, `mcp_server/server.py`.
- Políticas: `app/resilience.py`.
- Testes: `tests/unit/test_agent.py`, `tests/unit/test_coordinator.py`,
  `tests/contracts/test_mcp.py`, `tests/unit/test_resilience.py`.
- Comandos: `.venv/bin/python -m pytest tests/unit/test_agent.py tests/unit/test_coordinator.py tests/contracts/test_mcp.py tests/unit/test_resilience.py`; `make eval-offline`.

## Limite da evidência

O comparativo offline executa os agentes ADK sem credenciais externas. Isso
valida transferência, ferramentas e contratos locais; a comparação live continua
dependente de credenciais Gemini/Vertex e não foi executada por esse gate.
