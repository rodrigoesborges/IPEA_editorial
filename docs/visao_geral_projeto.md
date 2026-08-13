# IPEA Editorial — Visão Geral do Projeto

## 1) Natureza e Estrutura do Projeto

O **IPEA Editorial** é um sistema de **revisão editorial automatizada** para documentos acadêmicos (.docx, .pdf). Ele foi construído para o IPEA (Instituto de Pesquisa Econômica Aplicada) e faz exatamente o que um revisor humano faria: lê o texto inteiro, identifica problemas de gramática, tipografia, tabelas, figuras, **referências bibliográficas**, e devolve o documento com **comentários orientados (balões do Word)** apontando cada erro e sugerindo a correção.

### Como o pipeline funciona (4 camadas)

```
Documento de entrada (.docx/.pdf)
        │
        ▼
┌─ 1. Carga ──────────────────────────────────────┐
│  document_loader.py                              │
│  Lê o arquivo, extrai parágrafos, classifica     │
│  cada bloco (cabeçalho, corpo, referência, etc.) │
│  → gera um "NormalizedDocument"                  │
└──────────────────────────────────────────────────┘
        │
        ▼
┌─ 2. Escopo ─────────────────────────────────────┐
│  pipeline/scope.py                               │
│  Decide quais parágrafos cada agente vai ver     │
│  (ex.: agente de referências só vê a lista final │
│   + citações no corpo, não vê a bibliografia)    │
└──────────────────────────────────────────────────┘
        │
        ▼
┌─ 3. Execução ───────────────────────────────────┐
│  agents/ + prompts/ + LLM (modelos de IA)        │
│  6 agentes rodam em paralelo (até 3 por vez):    │
│                                                   │
│  • sinopse_abstract  (sin) — resumo/abstract      │
│  • gramatica_ortografia (gram) — gramática        │
│  • coerencia_logica (log) — coerência [experimental]│
│  • tabelas_figuras (tab) — tabelas e figuras      │
│  • referencias (ref) — citações e referências     │
│  • tipografia (tip) — tipo, negrito, itálico...   │
│                                                   │
│  Cada agente:                                     │
│    1. Roda heurísticas determinísticas (regex,    │
│       regras ABNT) — sem IA, 100% reprodutível    │
│    2. Chama o LLM para análises mais complexas    │
│    3. Valida/filtra resultados (descarta lixo)    │
└──────────────────────────────────────────────────┘
        │
        ▼
┌─ 4. Consolidação ───────────────────────────────┐
│  pipeline/consolidation.py + coordinator.py      │
│  Junta tudo, remove duplicatas, ordena comentários│
│  → exporta DOCX com balões + relatório JSON       │
└──────────────────────────────────────────────────┘
```

### A camada de referências bibliográficas (nosso foco)

É a parte mais complexa do sistema, separada em três etapas:

| Etapa | Módulo | O que faz |
|---|---|---|
| Extração | `abnt_citation_parser.py` | Encontra citações no corpo do texto: `(Silva, 2023)`, `Silva (2023)`, etc. |
| Pareamento | `abnt_matcher.py` | Compara cada citação com cada referência da lista final |
| Validação | `abnt_validator.py` | Verifica se cada referência segue as regras da ABNT (NBR 6023) |

O resultado é um objeto chamado `ReferencePipelineArtifact` (em `models.py`) que contém: citações encontradas, referências encontradas, âncoras exatas/prováveis, citações órfãs (sem referência) e referências órfãs (sem citação).

### Onde moram os arquivos

```
src/editorial_docx/
├── __main__.py          ← CLI (linha de comando)
├── graph_chat.py        ← fachada de orquestração (789 linhas)
├── config.py            ← configuração de runtime
├── models.py            ← estruturas de dados centrais
├── document_loader.py   ← leitura de .docx/.pdf
├── docx_utils.py        ← escrita de .docx com comentários (1.467 linhas!)
├── abnt_*.py            ← 7 módulos de regras bibliográficas ABNT
├── citations_eval.py    ← nosso pipeline de teste (novo)
├── pipeline/            ← orquestração, escopo, validação
├── agents/              ← heurísticas + validação por agente
├── prompts/             ← instruções de sistema de cada agente
└── references/          ← shims (apelidos) dos módulos abnt_*

streamlit_app.py         ← interface web (1.155 linhas)
testes/                  ← testes + dataset ouro
pyproject.toml           ← configuração do projeto Python
```

---

## 2) Fraquezas Principais

O projeto é funcional e está em produção, mas tem problemas estruturais conhecidos (inclusive documentados pelos próprios autores):

### Crítico: testes não estão no git

O `.gitignore` contém `/testes` — **a pasta inteira de testes é ignorada pelo Git**. Isso significa que os testes não são versionados, não rodam em CI, e mudanças de código podem quebrar coisas sem ninguém perceber. O mesmo vale para `*.json` e `*.docx` (globais).

### Crítico: o "dataset ouro" está vazio

Existe toda a ferramenta para medir qualidade (`gold_dataset.py`, `gold_metrics.py`), mas `metricas_reais.json` mostra **zero anotações humanas reais** — todas as métricas são 0.0. Não há baseline para detectar se uma mudança de prompt piorou ou melhorou o sistema.

### Arquitetura duplicada

| Problema | Detalhe |
|---|---|
| **Dois orquestradores** | `graph_chat.py` (789 linhas) e `pipeline/orchestrator.py` (326 linhas) fazem quase a mesma coisa |
| **11 arquivos shim** | As pastas `references/` e `io/` são só `from ..X import *` — espelhos dos módulos `abnt_*.py` e dos arquivos raiz. Dois caminhos de import para o mesmo código |
| **Funções duplicadas** | `pipeline/runtime.py` define `_parse_comments_with_status` duas vezes (linhas 547 e 699) com strings diferentes (`"sem conteúdo"` vs `"sem conteudo"`) |

### Sem eval automatizado

O roadmap interno lista 7 melhorias arquiteturais (async, tracing, structured outputs, etc.) — **todas marcadas como `pending`**. Mudanças de prompt acontecem "no escuro", sem portão de qualidade.

### Monolito de teste

`test_graph_chat.py` tem **4.715 linhas** num arquivo só — impraticável de manter.

### Documentação com paths Windows

Os `.md` em `docs/` têm caminhos hardcoded (`D:\github\lang_IPEA_editorial\...`) que não funcionam no Linux.

---

## 3) Nossa Primeira Tarefa: Pipeline de Teste para Referências

### O problema que resolvemos

O agente `referencias` usa um LLM para encontrar inconsistências entre citações e referências. Mas **como saber se o LLM está certo?** Sem um padrão-ouro (ground truth), é impossível medir precisão, recall, ou detectar regressões.

### A ideia central

> **Comparar duas versões do mesmo documento** (antes e depois da revisão humana) e derivar o ground truth automaticamente.

```
Documento PRÉ-edição          Documento PÓS-edição
      │                              │
      ▼                              ▼
 Motor determinístico           Motor determinístico
 (parsers ABNT + matcher)       (parsers ABNT + matcher)
      │                              │
      ▼                              ▼
 Snapshot A                      Snapshot B
 {citações, referências,         {citações, referências,
  problemas encontrados}          problemas encontrados}
      │                              │
      └──────────┬───────────────────┘
                 ▼
          diff_snapshots()
                 │
         ┌───────┴───────┐
         ▼               ▼
   GOLD POSITIVES   GOLD NEGATIVES
   (problemas que    (problemas que
    o editor          persistiram —
    resolveu)         o editor decidiu
                      que não eram erro)
```

**Intuição:** se um problema aparece na versão pré-editada mas desaparece na pós-editada, o editor o resolveu → era um erro real (**gold positive**). Se aparece nas duas versões, o editor o manteve → não era erro, ou era algo aceitável (**gold negative**).

### O pipeline que construímos: `citations_eval.py`

O módulo implementa 5 estágios:

| Estágio | Função | O que faz |
|---|---|---|
| **0** | `analyze_loaded_document()` | Roda os mesmos parsers ABNT do sistema real sobre um documento, produzindo um snapshot com citações, referências e problemas |
| **1** | `diff_snapshots()` | Compara pré vs pós e deriva gold positives / gold negatives |
| **3** | `classify_llm_comments()` | Pega o relatório do LLM e classifica cada comentário: **TP** (acertou), **FP** (falso positivo), **FN** (perdeu um erro real), **unmatched** (não casou com nada) |
| **5a** | `build_diff_json()` | Gera JSON com métricas: precision, recall, F1, recall por tipo |
| **5b** | `build_gold_dataset()` | Gera dataset no formato do `gold_metrics.py` para uso futuro em CI |

### Classificação dos comentários do LLM

```
Comentário do LLM
      │
      ▼
 Extrai (autor, ano) do texto do comentário
 (via extract_citation_candidates / parse_reference_entry)
      │
      ▼
 Adivinha o tipo: "missing_citation"? "uncited_reference"? "probable_match"?
      │
      ▼
 Compara com o gold diff:
   ┌─ Está em gold_positives? → TRUE POSITIVE ✓
   ├─ Está em gold_negatives? → FALSE POSITIVE (persistiu) ✗
   ├─ Está no snapshot determinístico? → LLM confirmou heurística
   └─ Não está em nenhum?     → FALSE POSITIVE (novel) ✗
```

### As métricas que calculamos

| Métrica | Fórmula | Significado |
|---|---|---|
| **Precision** | TP / (TP + FP) | Dos que o LLM flagrou, quantos eram reais? |
| **Recall** | TP / (TP + FN) | Dos reais, quantos o LLM flagrou? |
| **F1** | 2·P·R/(P+R) | Média harmônica |
| **Recall_missing** | detectados / gold_missing | Recall só para citações sem referência |
| **Recall_uncited** | detectados / gold_uncited | Recall só para referências sem citação |

### Como usar

```bash
# Linha de comando
uv run editorial-citations-eval \
  --original documento_pre.docx \
  --final documento_pos.docx \
  --report relatorio_llm.json \
  --output diff.json \
  --gold-output dataset_ouro.json \
  --model-name gpt-5.2 \
  --run-label "exp-001"
```

```python
# Em código Python
from editorial_docx.citations_eval import run_citation_eval
resultado = run_citation_eval(
    Path("pre.docx"), Path("pos.docx"),
    report_path=Path("relatorio.json"),
)
print(f"Precision: {resultado.metrics.precision:.2%}")
print(f"Recall:    {resultado.metrics.recall:.2%}")
```

### Cobertura de testes

Criamos 13 testes em `testes/test_citations_pipeline.py` que rodam offline (sem LLM, sem rede), usando documentos sintéticos. Todos passam:

| Teste | O que verifica |
|---|---|
| `test_analyze_loaded_document_finds_missing_and_uncited` | Detecta citação sem ref e ref sem citação |
| `test_analyze_loaded_document_finds_exact_match_silva` | Confirma match exato Silva |
| `test_diff_snapshots_resolved_vs_persisted` | Diff separa resolvidos de persistidos |
| `test_canonical_citation_key_from_text_extracts_narrative` | Extrai chave de citação narrativa |
| `test_canonical_citation_key_from_text_extracts_reference` | Extrai chave de referência ABNT |
| `test_canonical_citation_key_from_text_returns_none_for_garbage` | Rejeita texto sem citação |
| `test_extract_citation_comments_from_report_filters` | Filtra só comentários de referência |
| `test_classify_llm_comments_matches_gold_positive` | Classifica TP corretamente |
| `test_classify_llm_comments_flags_fp_persisted` | Classifica FP persistido |
| `test_build_gold_dataset_schema_matches_taxonomy` | Schema do dataset ouro válido |
| `test_build_diff_json_three_blocks_present` | JSON de diff tem todos os blocos |
| `test_run_citation_eval_end_to_end_synthetic` | Pipeline ponta-a-ponta funciona |
| `test_pipeline_artifact_conversion_round_trip` | Conversão para artifact preserva dados |

### Por que isso importa

Este pipeline é o **primeiro passo** para ter um eval automatizado: rode o agente de referências em um documento, compare com o que o editor realmente fez, e meça objetivamente se o LLM está melhorando ou piorando a cada mudança de prompt. Sem isso, qualquer ajuste é uma aposta cega.
