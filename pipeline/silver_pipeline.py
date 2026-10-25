from pyspark import pipelines as dp
from pyspark.sql import functions as F

spark.conf.set(
    "spark.sql.session.timeZone",
    "America/Sao_Paulo"
)


def bronze(nome):
    """
    Lê uma materialized view da camada Bronze.
    """
    return spark.read.table(nome)


def minutos_entre(fim, inicio):
    """
    Calcula a diferença entre timestamps em minutos.
    """
    return (
        F.unix_timestamp(fim)
        - F.unix_timestamp(inicio)
    ) / 60.0


def dim_tecnico():
    return (
        bronze("bronze_dim_tecnico")
        .dropDuplicates(["tecnico_id"])
        .select(
            "tecnico_id",
            F.col("nome_completo").alias("tecnico_nome"),
            F.col("cargo").alias("tecnico_cargo")
        )
    )


def dim_instituicao():
    return (
        bronze("bronze_dim_instituicao")
        .dropDuplicates(["instituicao_id"])
        .select(
            "instituicao_id",
            F.col("nome").alias("instituicao_nome")
        )
    )


def dim_unidade():
    unidades = (
        bronze("bronze_dim_unidade")
        .dropDuplicates(["unidade_id"])
        .select(
            "unidade_id",
            "instituicao_id",
            F.col("nome").alias("unidade_nome"),
            "cidade",
            "estado"
        )
    )

    return (
        unidades
        .join(
            dim_instituicao(),
            "instituicao_id",
            "left"
        )
    )


def dim_local():
    return (
        bronze("bronze_dim_local")
        .dropDuplicates(["local_id"])
        .select(
            "local_id",
            "nome_local",
            "tipo_local"
        )
    )


def dim_categoria_problema():
    return (
        bronze("bronze_dim_categoria_problema")
        .dropDuplicates(["categoria_problema_id"])
        .select(
            "categoria_problema_id",
            F.col("nome").alias("categoria_problema")
        )
    )


def dim_categoria_equipamento():
    return (
        bronze("bronze_dim_categoria_equipamento")
        .dropDuplicates(["categoria_equipamento_id"])
        .select(
            "categoria_equipamento_id",
            F.col("nome").alias("categoria_equipamento")
        )
    )


def dim_status_os():
    return (
        bronze("bronze_dim_status_ordem_servico")
        .dropDuplicates(["status_id"])
        .select(
            "status_id",
            F.col("nome").alias("status_os")
        )
    )


def dim_status_problema():
    return (
        bronze("bronze_dim_status_problema")
        .dropDuplicates(["status_problema_id"])
        .select(
            "status_problema_id",
            F.col("nome").alias("status_problema")
        )
    )


def tarefas_por_os():
    """
    Agrega tarefas por ordem de serviço para o KPI OS-15.
    """
    return (
        bronze("bronze_tarefa")
        .select(
            "ordem_servico_id",
            F.col("status_ordem_servico")
            .cast("int")
            .alias("status_tarefa")
        )
        .groupBy("ordem_servico_id")
        .agg(
            F.count("*")
            .alias("qtd_tarefas_origem"),

            F.sum(
                F.when(
                    F.col("status_tarefa") == 4,
                    1
                ).otherwise(0)
            ).alias("qtd_tarefas_concluidas")
        )
    )


@dp.materialized_view(
    name="silver_ordem_servico"
)
def silver_ordem_servico():

    df = (
        bronze("bronze_fato_ordem_servico")
        .dropDuplicates(["ordem_servico_id"])
    )

    for coluna in [
        "data_criacao",
        "data_prevista",
        "data_inicio",
        "data_conclusao"
    ]:
        df = df.withColumn(
            coluna,
            F.col(coluna).cast("timestamp")
        )

    df = (
        df
        .withColumn(
            "status_id",
            F.col("status_id").cast("int")
        )
        .withColumn(
            "qtd_tarefas",
            F.col("qtd_tarefas").cast("long")
        )
        .withColumn(
            "tempo_pausado_minutos",
            F.col("tempo_pausado_minutos").cast("double")
        )
        .withColumn(
            "tempo_resolucao_minutos",
            F.col("tempo_resolucao_minutos").cast("double")
        )
    )

    # Tempos negativos são tratados como inválidos
    df = (
        df
        .withColumn(
            "tempo_pausado_minutos",
            F.when(
                F.col("tempo_pausado_minutos") >= 0,
                F.col("tempo_pausado_minutos")
            )
        )
        .withColumn(
            "tempo_resolucao_minutos",
            F.when(
                F.col("tempo_resolucao_minutos") >= 0,
                F.col("tempo_resolucao_minutos")
            )
        )
    )

    # Regras de status usadas nos KPIs
    df = (
        df
        .withColumn(
            "esta_aberta",
            F.col("status_id").isin(1, 2, 3)
        )
        .withColumn(
            "finalizada",
            F.col("status_id") == 4
        )
    )

    espera_valida = (
        F.col("data_criacao").isNotNull()
        & F.col("data_inicio").isNotNull()
        & (
            F.col("data_inicio")
            >= F.col("data_criacao")
        )
    )

    df = (
        df
        .withColumn(
            "espera_valida",
            espera_valida
        )
        .withColumn(
            "tempo_espera_minutos",
            F.when(
                espera_valida,
                minutos_entre(
                    F.col("data_inicio"),
                    F.col("data_criacao")
                )
            )
        )
    )

    datas_prazo_validas = (
        F.col("data_criacao").isNotNull()
        & F.col("data_prevista").isNotNull()
        & F.col("data_conclusao").isNotNull()
        & (
            F.col("data_conclusao")
            >= F.col("data_criacao")
        )
    )

    df = (
        df
        .withColumn(
            "datas_prazo_validas",
            datas_prazo_validas
        )
        .withColumn(
            "concluida_no_prazo_calc",
            F.when(
                datas_prazo_validas,
                F.col("data_conclusao")
                <= F.col("data_prevista")
            )
        )
    )

    return (
        df

        .join(
            tarefas_por_os(),
            "ordem_servico_id",
            "left"
        )

        .join(
            dim_tecnico(),
            "tecnico_id",
            "left"
        )

        .join(
            dim_status_os(),
            "status_id",
            "left"
        )

        .join(
            dim_categoria_problema(),
            "categoria_problema_id",
            "left"
        )

        .join(
            dim_unidade(),
            "unidade_id",
            "left"
        )

        .join(
            dim_local(),
            "local_id",
            "left"
        )

        .withColumn(
            "data_criacao_dia",
            F.to_date("data_criacao")
        )

        .withColumn(
            "data_conclusao_dia",
            F.to_date("data_conclusao")
        )
    )


@dp.materialized_view(
    name="silver_problema"
)
def silver_problema():

    df = (
        bronze("bronze_fato_problema")
        .dropDuplicates(["problema_id"])
        .withColumn(
            "data_criacao",
            F.col("data_criacao").cast("timestamp")
        )
        .withColumn(
            "status_problema_id",
            F.col("status_problema_id").cast("int")
        )
    )

    df = (
        df
        .withColumn(
            "esta_pendente",
            F.col("status_problema_id") == 0
        )
        .withColumn(
            "esta_reprovado",
            F.col("status_problema_id") == 1
        )
        .withColumn(
            "esta_aprovado",
            F.col("status_problema_id") == 2
        )
        .withColumn(
            "gerou_ordem_servico",
            F.col("gerou_ordem_servico").cast("boolean")
        )
        .withColumn(
            "data_criacao_dia",
            F.to_date("data_criacao")
        )
    )

    return (
        df

        .join(
            dim_status_problema(),
            "status_problema_id",
            "left"
        )

        .join(
            dim_categoria_equipamento(),
            "categoria_equipamento_id",
            "left"
        )

        .join(
            dim_unidade(),
            "unidade_id",
            "left"
        )

        .join(
            dim_local(),
            "local_id",
            "left"
        )
    )


@dp.materialized_view(
    name="silver_ocorrencia"
)
def silver_ocorrencia():

    df = (
        bronze("bronze_fato_ocorrencia")
        .dropDuplicates(["ocorrencia_id"])

        .withColumn(
            "data_criacao",
            F.col("data_criacao").cast("timestamp")
        )

        .withColumn(
            "prioridade",
            F.col("prioridade").cast("int")
        )

        .withColumn(
            "esta_ativo",
            F.col("esta_ativo").cast("boolean")
        )

        .withColumn(
            "data_criacao_dia",
            F.to_date("data_criacao")
        )
    )

    # Padroniza os nomes das prioridades
    df = (
        df
        .withColumn(
            "prioridade_nome",
            F.when(
                F.col("prioridade") == 0,
                "Alta"
            )
            .when(
                F.col("prioridade") == 1,
                "Média"
            )
            .when(
                F.col("prioridade") == 2,
                "Baixa"
            )
            .otherwise("Não informada")
        )
    )

    return (
        df

        .join(
            dim_tecnico(),
            "tecnico_id",
            "left"
        )

        .join(
            dim_categoria_problema(),
            "categoria_problema_id",
            "left"
        )

        .join(
            dim_categoria_equipamento(),
            "categoria_equipamento_id",
            "left"
        )

        .join(
            dim_unidade(),
            "unidade_id",
            "left"
        )

        .join(
            dim_local(),
            "local_id",
            "left"
        )
    )