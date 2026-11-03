from datetime import datetime
from pathlib import Path
import subprocess

from pyspark.sql import functions as F


# TROQUE SOMENTE PELO CAMINHO REAL DO SEU GIT FOLDER.
GIT_FOLDER = "/Workspace/Users/fixai.inter2026@gmail.com/segundo-business-intelligence"

BRANCH = "dev"
UNIDADE_EXPORTADA = 11
TABELA_ORIGEM = "workspace.gold.dashboard_os_criacao"
ARQUIVO_RELATIVO = "exports/gold/kpi_os_criacao_diario_unidade_11.csv"


def executar_git(*argumentos, check=True):
    resultado = subprocess.run(
        ["git", *argumentos],
        cwd=GIT_FOLDER,
        capture_output=True,
        text=True,
        check=False,
    )

    if check and resultado.returncode != 0:
        raise RuntimeError(
            f"Falha em: git {' '.join(argumentos)}\n"
            f"stdout:\n{resultado.stdout}\n"
            f"stderr:\n{resultado.stderr}"
        )

    return resultado


# COMMAND ----------

# Valida o Git folder e impede push acidental para outra branch.
git_path = Path(GIT_FOLDER)

if not git_path.exists():
    raise FileNotFoundError(
        f"Git folder não encontrado: {GIT_FOLDER}. "
        "Copie o caminho completo da pasta Git no Databricks Workspace."
    )

if not (git_path / ".git").exists():
    raise RuntimeError(f"A pasta {GIT_FOLDER} não é um Git folder clonado.")

branch_atual = executar_git("branch", "--show-current").stdout.strip()

if branch_atual != BRANCH:
    raise RuntimeError(
        f"Branch atual: {branch_atual!r}. "
        f"Selecione a branch {BRANCH!r} no Git folder antes de executar."
    )

print(f"Git folder validado: {GIT_FOLDER}")
print(f"Branch validada: {branch_atual}")


# COMMAND ----------

# Lê a tabela detalhada já restrita e aplica novamente a Unidade 11
# como defesa adicional antes da exportação.
df_origem = spark.table(TABELA_ORIGEM)

colunas_obrigatorias = {
    "unidade_id",
    "data_referencia",
    "qtd_os",
    "tempo_espera_minutos",
    "tempo_pausado_minutos",
    "reabertura_pct_item",
    "reatribuicao_pct_item",
}

colunas_ausentes = sorted(colunas_obrigatorias - set(df_origem.columns))

if colunas_ausentes:
    raise RuntimeError(
        f"Colunas ausentes em {TABELA_ORIGEM}: {colunas_ausentes}"
    )

df_unidade = df_origem.filter(
    F.col("unidade_id").cast("long") == F.lit(UNIDADE_EXPORTADA)
)

if df_unidade.limit(1).count() == 0:
    raise RuntimeError(
        f"Nenhum registro da Unidade {UNIDADE_EXPORTADA} foi encontrado "
        f"em {TABELA_ORIGEM}."
    )

# Garante que o dataframe de saída não carregue outra unidade.
unidades_encontradas = [
    linha.unidade_id
    for linha in df_unidade.select(
        F.col("unidade_id").cast("long").alias("unidade_id")
    )
    .distinct()
    .collect()
]

if unidades_encontradas != [UNIDADE_EXPORTADA]:
    raise RuntimeError(
        f"Validação de segurança falhou. Unidades encontradas: "
        f"{unidades_encontradas}"
    )

print(f"Unidade validada para exportação: {UNIDADE_EXPORTADA}")


# COMMAND ----------

# Agrega antes de exportar. Nenhuma OS individual vai para o GitHub.
df_exportacao = (
    df_unidade
    .groupBy("data_referencia")
    .agg(
        F.sum("qtd_os").cast("long").alias("qtd_os_criadas"),
        F.round(F.avg("tempo_espera_minutos"), 2).alias(
            "tempo_medio_espera_min"
        ),
        F.round(F.avg("tempo_pausado_minutos"), 2).alias(
            "tempo_medio_pausado_min"
        ),
        F.round(F.avg("reabertura_pct_item"), 2).alias(
            "taxa_reabertura_pct"
        ),
        F.round(F.avg("reatribuicao_pct_item"), 2).alias(
            "taxa_reatribuicao_pct"
        ),
    )
    .withColumn("unidade_id", F.lit(UNIDADE_EXPORTADA).cast("long"))
    .select(
        "unidade_id",
        "data_referencia",
        "qtd_os_criadas",
        "tempo_medio_espera_min",
        "tempo_medio_pausado_min",
        "taxa_reabertura_pct",
        "taxa_reatribuicao_pct",
    )
    .orderBy("data_referencia")
)

display(df_exportacao)


# COMMAND ----------

# Grava um único CSV agregado dentro do Git folder.
csv_path = git_path / ARQUIVO_RELATIVO
csv_path.parent.mkdir(parents=True, exist_ok=True)

df_exportacao.toPandas().to_csv(
    csv_path,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    na_rep="",
)

print(f"CSV gerado em: {csv_path}")


# COMMAND ----------

# Versiona somente o CSV seguro e envia para a branch dev.
executar_git("add", ARQUIVO_RELATIVO)

tem_alteracoes = executar_git(
    "diff", "--cached", "--quiet", check=False
).returncode != 0

if not tem_alteracoes:
    print("O CSV não mudou. Commit e push não são necessários.")
    dbutils.notebook.exit("Sem alterações no CSV da Unidade 11")

commit_msg = (
    f"Atualiza KPI Gold da Unidade {UNIDADE_EXPORTADA} em "
    + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
)

resultado_commit = executar_git("commit", "-m", commit_msg)
print(resultado_commit.stdout)

resultado_push = executar_git("push", "origin", BRANCH)
print(resultado_push.stdout)
print(resultado_push.stderr)

print(
    f"CSV agregado da Unidade {UNIDADE_EXPORTADA} enviado "
    f"com sucesso para a branch {BRANCH!r}."
)
