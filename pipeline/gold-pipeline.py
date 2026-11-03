from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd

spark.conf.set("spark.sql.session.timeZone", "America/Sao_Paulo")

def tabela(nome):
    """Lê uma tabela/view do mesmo schema-alvo e mantém a dependência no grafo."""
    return spark.read.table(nome)


def percentual(numerador, denominador):
    """Percentual agregado; retorna NULL quando não existe população."""
    return F.when(
        denominador > 0,
        F.round(numerador.cast("double") * F.lit(100.0) / denominador, 2),
    )


def indicador_pct(condicao, populacao=None):
    """100/0 por entidade; fora da população opcional retorna NULL."""
    valor = F.when(condicao, F.lit(100.0)).otherwise(F.lit(0.0))
    if populacao is None:
        return valor
    return F.when(populacao, valor)

DIMENSOES_OS = [
    "tecnico_id",
    "tecnico_nome",
    "unidade_id",
    "unidade_nome",
    "instituicao_id",
    "instituicao_nome",
    "cidade",
    "estado",
    "local_id",
    "nome_local",
    "tipo_local",
    "categoria_problema_id",
    "categoria_problema",
    "status_id",
    "status_os",
]

JANELA_EPISODIO = pd.Timedelta(hours=24)
JANELA_RECORRENCIA = pd.Timedelta(days=30)

SCHEMA_EPISODIOS = """
    unidade_id long,
    local_id string,
    categoria_equipamento_id string,
    inicio_episodio timestamp,
    fim_episodio timestamp,
    qtd_solicitacoes long,
    recorrente boolean
"""


def consolidar_episodios(pdf):
    pdf = pdf.sort_values(["data_criacao", "problema_id"])

    unidade_id = int(pdf["unidade_id"].iloc[0])
    local_id = str(pdf["local_id"].iloc[0])
    categoria_id = str(pdf["categoria_equipamento_id"].iloc[0])

    episodios = []
    inicio_episodio = None

    for _, registro in pdf.iterrows():
        data = registro["data_criacao"]

        if pd.isna(data):
            continue

        if (
            inicio_episodio is None
            or data >= inicio_episodio + JANELA_EPISODIO
        ):
            inicio_episodio = data

            episodios.append(
                {
                    "inicio_episodio": data,
                    "fim_episodio": data,
                    "qtd_solicitacoes": 0,
                }
            )

        episodios[-1]["fim_episodio"] = data
        episodios[-1]["qtd_solicitacoes"] += 1

    resultado = []
    inicio_anterior = None

    for episodio in episodios:
        inicio_atual = episodio["inicio_episodio"]

        recorrente = (
            inicio_anterior is not None
            and inicio_atual > inicio_anterior
            and inicio_atual <= inicio_anterior + JANELA_RECORRENCIA
        )

        resultado.append(
            {
                "unidade_id": unidade_id,
                "local_id": local_id,
                "categoria_equipamento_id": categoria_id,
                "inicio_episodio": inicio_atual,
                "fim_episodio": episodio["fim_episodio"],
                "qtd_solicitacoes": episodio["qtd_solicitacoes"],
                "recorrente": recorrente,
            }
        )

        inicio_anterior = inicio_atual

    return pd.DataFrame(resultado)

# ============================================================
# GOLD AGREGADA — AUDITORIA, CONFERÊNCIA E VALIDAÇÃO
# ============================================================


@dp.materialized_view(name="gold_kpi_os_criacao_diario")
def gold_kpi_os_criacao_diario():
    df = tabela("silver_ordem_servico")
    return (
        df.groupBy(
            "unidade_id",
            F.col("data_criacao_dia").alias("data_referencia"),
        )
        .agg(
            F.countDistinct("ordem_servico_id").alias("qtd_os_criadas"),
            F.avg("tempo_espera_minutos").alias("tempo_medio_espera_min"),
            F.avg("tempo_pausado_minutos").alias("tempo_medio_pausado_min"),
            F.avg("qtd_tarefas").alias("media_tarefas_por_os"),
            F.sum(F.when(F.col("foi_reaberta") == True, 1).otherwise(0)).alias(
                "qtd_os_reabertas"
            ),
            F.sum(F.when(F.col("foi_reatribuida") == True, 1).otherwise(0)).alias(
                "qtd_os_reatribuidas"
            ),
        )
        .withColumn(
            "taxa_reabertura_pct",
            percentual(F.col("qtd_os_reabertas"), F.col("qtd_os_criadas")),
        )
        .withColumn(
            "taxa_reatribuicao_pct",
            percentual(F.col("qtd_os_reatribuidas"), F.col("qtd_os_criadas")),
        )
    )


@dp.materialized_view(name="gold_kpi_os_conclusao_diario")
def gold_kpi_os_conclusao_diario():
    df = tabela("silver_ordem_servico").filter(
        F.col("finalizada") & F.col("data_conclusao").isNotNull()
    )
    return (
        df.groupBy(
            "unidade_id",
            F.col("data_conclusao_dia").alias("data_referencia"),
        )
        .agg(
            F.countDistinct("ordem_servico_id").alias("qtd_os_finalizadas"),
            F.sum(F.when(F.col("datas_prazo_validas") == True, 1).otherwise(0)).alias(
                "qtd_os_com_prazo_valido"
            ),
            F.sum(
                F.when(
                    (F.col("datas_prazo_validas") == True)
                    & (F.col("concluida_no_prazo_calc") == True),
                    1,
                ).otherwise(0)
            ).alias("qtd_os_concluidas_no_prazo"),
            F.avg("tempo_resolucao_minutos").alias("tempo_medio_resolucao_min"),
        )
        .withColumn(
            "taxa_conclusao_prazo_pct",
            percentual(
                F.col("qtd_os_concluidas_no_prazo"),
                F.col("qtd_os_com_prazo_valido"),
            ),
        )
    )


@dp.materialized_view(name="gold_kpi_os_situacao_atual")
def gold_kpi_os_situacao_atual():
    df = tabela("silver_ordem_servico")
    aberta = F.col("status_id").isin(1, 2, 3)
    return (
        df.groupBy("unidade_id").agg(
            F.countDistinct("ordem_servico_id").alias("qtd_os_total"),
            F.sum(F.when(aberta, 1).otherwise(0)).alias("qtd_os_abertas"),
            F.sum(F.when(F.col("status_id") == 1, 1).otherwise(0)).alias(
                "qtd_os_atrasadas"
            ),
            F.sum(
                F.when(
                    aberta
                    & F.col("data_prevista").isNotNull()
                    & (F.col("data_prevista") < F.current_timestamp()),
                    1,
                ).otherwise(0)
            ).alias("qtd_os_prazo_vencido"),
        )
        .withColumn(
            "taxa_atraso_pct",
            percentual(F.col("qtd_os_atrasadas"), F.col("qtd_os_abertas")),
        )
        .withColumn("data_referencia_snapshot", F.current_timestamp())
    )


@dp.materialized_view(name="gold_kpi_carga_ativa_tecnico")
def gold_kpi_carga_ativa_tecnico():
    df = tabela("silver_ordem_servico").filter(F.col("status_id").isin(1, 2, 3))
    return (
        df.groupBy("tecnico_id", "tecnico_nome","unidade_id")
        .agg(
            F.countDistinct("ordem_servico_id").alias("qtd_os_ativas"),
            F.sum(F.when(F.col("status_id") == 1, 1).otherwise(0)).alias(
                "qtd_os_atrasadas"
            ),
            F.sum(
                F.when(
                    F.col("data_prevista").isNotNull()
                    & (F.col("data_prevista") < F.current_timestamp()),
                    1,
                ).otherwise(0)
            ).alias("qtd_os_prazo_vencido"),
        )
        .withColumn(
            "taxa_atraso_pct",
            percentual(F.col("qtd_os_atrasadas"), F.col("qtd_os_ativas")),
        )
    )


@dp.materialized_view(name="gold_kpi_produtividade_tecnico")
def gold_kpi_produtividade_tecnico():
    df = tabela("silver_ordem_servico").filter(
        F.col("finalizada") & F.col("data_conclusao").isNotNull()
    )
    return (
        df.groupBy(
            "unidade_id",
            F.col("data_conclusao_dia").alias("data_referencia"),
            "tecnico_id",
            "tecnico_nome",
        )
        .agg(
            F.countDistinct("ordem_servico_id").alias("qtd_os_finalizadas"),
            F.avg("tempo_resolucao_minutos").alias("tempo_medio_resolucao_min"),
            F.sum(F.when(F.col("datas_prazo_validas") == True, 1).otherwise(0)).alias(
                "qtd_os_com_prazo_valido"
            ),
            F.sum(
                F.when(
                    (F.col("datas_prazo_validas") == True)
                    & (F.col("concluida_no_prazo_calc") == True),
                    1,
                ).otherwise(0)
            ).alias("qtd_os_concluidas_no_prazo"),
        )
        .withColumn(
            "taxa_conclusao_prazo_pct",
            percentual(
                F.col("qtd_os_concluidas_no_prazo"),
                F.col("qtd_os_com_prazo_valido"),
            ),
        )
    )


@dp.materialized_view(name="gold_os_por_status")
def gold_os_por_status():
    df = tabela("silver_ordem_servico")
    return df.groupBy("unidade_id", "status_id", "status_os").agg(
        F.countDistinct("ordem_servico_id").alias("qtd_os")
    )


@dp.materialized_view(name="gold_os_tempos")
def gold_os_tempos():
    df = tabela("silver_ordem_servico")
    return df.select(
        "ordem_servico_id",
        "data_criacao",
        "data_prevista",
        "data_inicio",
        "data_conclusao",
        *DIMENSOES_OS,
        "tempo_espera_minutos",
        "tempo_pausado_minutos",
        "tempo_resolucao_minutos",
        "datas_prazo_validas",
        "concluida_no_prazo_calc",
    )


@dp.materialized_view(name="gold_kpi_solicitacoes_diario")
def gold_kpi_solicitacoes_diario():
    df = tabela("silver_problema")
    return (
        df.groupBy(
            F.col("data_criacao_dia").alias("data_referencia"),
            "unidade_id",
            "local_id",
            "categoria_equipamento_id",
        )
        .agg(
            F.countDistinct("problema_id").alias("qtd_solicitacoes"),
            F.sum(F.when(F.col("esta_pendente") == True, 1).otherwise(0)).alias(
                "qtd_solicitacoes_pendentes"
            ),
            F.sum(F.when(F.col("esta_aprovado") == True, 1).otherwise(0)).alias(
                "qtd_solicitacoes_aprovadas"
            ),
            F.sum(F.when(F.col("esta_reprovado") == True, 1).otherwise(0)).alias(
                "qtd_solicitacoes_reprovadas"
            ),
            F.sum(
                F.when(F.col("gerou_ordem_servico") == True, 1).otherwise(0)
            ).alias("qtd_solicitacoes_com_os"),
        )
        .withColumn(
            "taxa_aprovacao_pct",
            percentual(
                F.col("qtd_solicitacoes_aprovadas"), F.col("qtd_solicitacoes")
            ),
        )
        .withColumn(
            "taxa_reprovacao_pct",
            percentual(
                F.col("qtd_solicitacoes_reprovadas"), F.col("qtd_solicitacoes")
            ),
        )
        .withColumn(
            "taxa_conversao_os_pct",
            percentual(
                F.col("qtd_solicitacoes_com_os"), F.col("qtd_solicitacoes")
            ),
        )
    )


@dp.materialized_view(name="gold_kpi_ocorrencias_diario")
def gold_kpi_ocorrencias_diario():
    df = tabela("silver_ocorrencia")
    return (
        df.groupBy(
            F.col("data_criacao_dia").alias("data_referencia"),
            "tecnico_id",
            "unidade_id",
            "local_id",
            "categoria_problema_id",
            "categoria_equipamento_id",
            "prioridade",
        )
        .agg(
            F.countDistinct("ocorrencia_id").alias("qtd_ocorrencias"),
            F.sum(F.when(F.col("esta_ativo") == True, 1).otherwise(0)).alias(
                "qtd_ocorrencias_ativas"
            ),
            F.sum(F.when(F.col("prioridade") == 0, 1).otherwise(0)).alias(
                "qtd_ocorrencias_alta_prioridade"
            ),
        )
        .withColumn(
            "taxa_alta_prioridade_pct",
            percentual(
                F.col("qtd_ocorrencias_alta_prioridade"),
                F.col("qtd_ocorrencias"),
            ),
        )
    )


@dp.materialized_view(name="gold_recorrencia_episodios")
def gold_recorrencia_episodios():
    """Cria episódios fixos de 24 horas e identifica recorrência em até 30 dias."""

    base = (
        tabela("silver_problema")
        .filter(
            F.col("data_criacao").isNotNull()
            & F.col("unidade_id").isNotNull()
            & F.col("local_id").isNotNull()
            & F.col("categoria_equipamento_id").isNotNull()
        )
        .select(
            F.col("unidade_id").cast("long").alias("unidade_id"),
            F.col("local_id").cast("string").alias("local_id"),
            F.col("categoria_equipamento_id")
            .cast("string")
            .alias("categoria_equipamento_id"),
            F.col("problema_id").cast("string").alias("problema_id"),
            "data_criacao",
        )
    )

    return (
        base.groupBy(
            "unidade_id",
            "local_id",
            "categoria_equipamento_id",
        )
        .applyInPandas(
            consolidar_episodios,
            schema=SCHEMA_EPISODIOS,
        )
    )

@dp.materialized_view(name="gold_kpi_recorrencia_diario")
def gold_kpi_recorrencia_diario():
    df = tabela("gold_recorrencia_episodios")
    return (
        df.groupBy(
            F.to_date("inicio_episodio").alias("data_referencia"),
            "unidade_id",
            "local_id",
            "categoria_equipamento_id",
        )
        .agg(
            F.sum("qtd_solicitacoes").alias("qtd_solicitacoes"),
            F.count(F.lit(1)).alias("qtd_episodios"),
            F.sum(F.when(F.col("recorrente") == True, 1).otherwise(0)).alias(
                "qtd_episodios_recorrentes"
            ),
        )
        .withColumn(
            "taxa_recorrencia_pct",
            percentual(
                F.col("qtd_episodios_recorrentes"), F.col("qtd_episodios")
            ),
        )
    )


# ============================================================
# GOLD DETALHADA — FONTE DO DATABRICKS AI/BI DASHBOARD
# ============================================================


@dp.materialized_view(name="gold_dashboard_os_criacao")
def gold_dashboard_os_criacao():
    df = tabela("silver_ordem_servico")
    return (
        df.select(
            "ordem_servico_id",
            F.col("data_criacao_dia").alias("data_referencia"),
            "data_criacao",
            "data_prevista",
            "data_inicio",
            "data_conclusao",
            *DIMENSOES_OS,
            "tempo_espera_minutos",
            "tempo_pausado_minutos",
            "qtd_tarefas",
            "foi_reaberta",
            "foi_reatribuida",
        )
        .withColumn("qtd_os", F.lit(1))
        .withColumn(
            "reabertura_pct_item", indicador_pct(F.col("foi_reaberta") == True)
        )
        .withColumn(
            "reatribuicao_pct_item",
            indicador_pct(F.col("foi_reatribuida") == True),
        )
    )


@dp.materialized_view(name="gold_dashboard_os_conclusao")
def gold_dashboard_os_conclusao():
    df = tabela("silver_ordem_servico").filter(
        F.col("finalizada") & F.col("data_conclusao").isNotNull()
    )
    return (
        df.select(
            "ordem_servico_id",
            F.col("data_conclusao_dia").alias("data_referencia"),
            "data_criacao",
            "data_prevista",
            "data_inicio",
            "data_conclusao",
            *DIMENSOES_OS,
            "datas_prazo_validas",
            "concluida_no_prazo_calc",
            "tempo_resolucao_minutos",
        )
        .withColumn("qtd_os_finalizadas", F.lit(1))
        .withColumn(
            "conclusao_prazo_pct_item",
            indicador_pct(
                F.col("concluida_no_prazo_calc") == True,
                F.col("datas_prazo_validas") == True,
            ),
        )
    )


@dp.materialized_view(name="gold_dashboard_os_atual")
def gold_dashboard_os_atual():
    df = tabela("silver_ordem_servico")
    aberta = F.col("status_id").isin(1, 2, 3)
    return (
        df.select(
            "ordem_servico_id",
            "data_criacao",
            "data_prevista",
            "data_inicio",
            "data_conclusao",
            *DIMENSOES_OS,
        )
        .withColumn("data_referencia_snapshot", F.current_timestamp())
        .withColumn("os_aberta", F.when(aberta, 1).otherwise(0))
        .withColumn(
            "os_atrasada", F.when(F.col("status_id") == 1, 1).otherwise(0)
        )
        .withColumn(
            "os_prazo_vencido",
            F.when(
                aberta
                & F.col("data_prevista").isNotNull()
                & (F.col("data_prevista") < F.current_timestamp()),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "atraso_pct_item",
            indicador_pct(F.col("status_id") == 1, aberta),
        )
    )


@dp.materialized_view(name="gold_dashboard_tarefa")
def gold_dashboard_tarefa():
    # Mantém uma linha para cada linha/tarefa da Bronze. Inclui todas as tarefas.
    tarefas = tabela("bronze_tarefa").select(
        "ordem_servico_id",
        F.col("status_ordem_servico").cast("int").alias("status_tarefa"),
    )
    ordens = tabela("silver_ordem_servico").select(
        "ordem_servico_id",
        F.col("data_criacao_dia").alias("data_referencia"),
        *DIMENSOES_OS,
    )
    return (
        tarefas.join(ordens, "ordem_servico_id", "inner")
        .withColumn("qtd_tarefas", F.lit(1))
        .withColumn(
            "tarefa_concluida",
            F.when(F.col("status_tarefa") == 4, 1).otherwise(0),
        )
        .withColumn(
            "tarefa_concluida_pct_item",
            indicador_pct(F.col("status_tarefa") == 4),
        )
    )


@dp.materialized_view(name="gold_dashboard_solicitacao")
def gold_dashboard_solicitacao():
    df = tabela("silver_problema")
    return (
        df.select(
            "problema_id",
            F.col("data_criacao_dia").alias("data_referencia"),
            "data_criacao",
            "status_problema_id",
            "status_problema",
            "unidade_id",
            "unidade_nome",
            "instituicao_id",
            "instituicao_nome",
            "cidade",
            "estado",
            "local_id",
            "nome_local",
            "tipo_local",
            "categoria_equipamento_id",
            "categoria_equipamento",
            "esta_pendente",
            "esta_aprovado",
            "esta_reprovado",
            "gerou_ordem_servico",
        )
        .withColumn("qtd_solicitacoes", F.lit(1))
        .withColumn(
            "solicitacao_pendente",
            F.when(F.col("esta_pendente") == True, 1).otherwise(0),
        )
        .withColumn(
            "solicitacao_aprovada",
            F.when(F.col("esta_aprovado") == True, 1).otherwise(0),
        )
        .withColumn(
            "solicitacao_reprovada",
            F.when(F.col("esta_reprovado") == True, 1).otherwise(0),
        )
        .withColumn(
            "solicitacao_com_os",
            F.when(F.col("gerou_ordem_servico") == True, 1).otherwise(0),
        )
        .withColumn(
            "aprovacao_pct_item", indicador_pct(F.col("esta_aprovado") == True)
        )
        .withColumn(
            "reprovacao_pct_item",
            indicador_pct(F.col("esta_reprovado") == True),
        )
        .withColumn(
            "conversao_os_pct_item",
            indicador_pct(F.col("gerou_ordem_servico") == True),
        )
    )


@dp.materialized_view(name="gold_dashboard_ocorrencia")
def gold_dashboard_ocorrencia():
    df = tabela("silver_ocorrencia")
    return (
        df.select(
            "ocorrencia_id",
            F.col("data_criacao_dia").alias("data_referencia"),
            "data_criacao",
            "tecnico_id",
            "tecnico_nome",
            "unidade_id",
            "unidade_nome",
            "instituicao_id",
            "instituicao_nome",
            "cidade",
            "estado",
            "local_id",
            "nome_local",
            "tipo_local",
            "categoria_problema_id",
            "categoria_problema",
            "categoria_equipamento_id",
            "categoria_equipamento",
            "prioridade",
            "prioridade_nome",
            "esta_ativo",
        )
        .withColumn("qtd_ocorrencias", F.lit(1))
        .withColumn(
            "ocorrencia_ativa",
            F.when(F.col("esta_ativo") == True, 1).otherwise(0),
        )
        .withColumn(
            "ocorrencia_alta_prioridade",
            F.when(F.col("prioridade") == 0, 1).otherwise(0),
        )
        .withColumn(
            "alta_prioridade_pct_item",
            indicador_pct(F.col("prioridade") == 0),
        )
    )


@dp.materialized_view(name="gold_dashboard_recorrencia")
def gold_dashboard_recorrencia():
    episodios = (tabela("gold_recorrencia_episodios"))
    dimensoes = (
        tabela("silver_problema")
        .select(
            F.col("unidade_id").cast("long").alias("unidade_id"),
            F.col("local_id").cast("string").alias("local_id"),
            F.col("categoria_equipamento_id")
            .cast("string")
            .alias("categoria_equipamento_id"),
            "nome_local",
            "tipo_local",
            "categoria_equipamento",
        )
        .dropDuplicates(
            ["unidade_id", "local_id", "categoria_equipamento_id"]
        )
    )
    return (
        episodios.join(
            dimensoes,
            ["unidade_id", "local_id", "categoria_equipamento_id"],
            "left",
        )
        .withColumn("data_referencia", F.to_date("inicio_episodio"))
        .withColumn("qtd_episodios", F.lit(1))
        .withColumn(
            "qtd_episodios_recorrentes",
            F.when(F.col("recorrente") == True, 1).otherwise(0),
        )
        .withColumn(
            "episodio_recorrente_pct_item",
            indicador_pct(F.col("recorrente") == True),
        )
    )
