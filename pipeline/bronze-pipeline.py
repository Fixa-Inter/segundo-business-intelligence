from pyspark import pipelines as dp
from pyspark.sql import functions as F

# Conexão com PostgreSQL
PG_HOST = "pg-3117a07b-fixa-9a44.e.aivencloud.com"
PG_PORT = "15976"
PG_DATABASE = "db2anoPrincipal"
PG_SCHEMA = "public"

PG_USER = dbutils.secrets.get(
    scope="fixa",
    key="postgres-user"
).strip()

PG_PASSWORD = dbutils.secrets.get(
    scope="fixa",
    key="postgres-password"
).strip()

JDBC_URL = (
    f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
    "?sslmode=require"
)


def ler_postgres(nome_objeto):
    """
    Lê uma tabela ou view do PostgreSQL via JDBC.
    """
    return (
        spark.read
        .format("jdbc")
        .option("url", JDBC_URL)
        .option(
            "dbtable",
            f"{PG_SCHEMA}.{nome_objeto}"
        )
        .option("user", PG_USER)
        .option("password", PG_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .option("fetchsize", "10000")
        .load()
    )


def adicionar_auditoria(df, nome_objeto):
    """
    Adiciona metadados técnicos da ingestão.
    """
    return (
        df
        .withColumn(
            "_ingestion_timestamp",
            F.current_timestamp()
        )
        .withColumn(
            "_source_system",
            F.lit("postgresql")
        )
        .withColumn(
            "_source_object",
            F.lit(nome_objeto)
        )
    )


@dp.materialized_view(
    name="bronze_dim_categoria_equipamento"
)
def bronze_dim_categoria_equipamento():
    return adicionar_auditoria(
        ler_postgres("dim_categoria_equipamento"),
        "dim_categoria_equipamento"
    )


@dp.materialized_view(
    name="bronze_dim_categoria_problema"
)
def bronze_dim_categoria_problema():
    return adicionar_auditoria(
        ler_postgres("dim_categoria_problema"),
        "dim_categoria_problema"
    )


@dp.materialized_view(
    name="bronze_dim_instituicao"
)
def bronze_dim_instituicao():
    return adicionar_auditoria(
        ler_postgres("dim_instituicao"),
        "dim_instituicao"
    )


@dp.materialized_view(
    name="bronze_dim_local"
)
def bronze_dim_local():
    return adicionar_auditoria(
        ler_postgres("dim_local"),
        "dim_local"
    )


@dp.materialized_view(
    name="bronze_dim_status_ordem_servico"
)
def bronze_dim_status_ordem_servico():
    return adicionar_auditoria(
        ler_postgres("dim_status_ordem_servico"),
        "dim_status_ordem_servico"
    )


@dp.materialized_view(
    name="bronze_dim_status_problema"
)
def bronze_dim_status_problema():
    return adicionar_auditoria(
        ler_postgres("dim_status_problema"),
        "dim_status_problema"
    )


@dp.materialized_view(
    name="bronze_dim_tecnico"
)
def bronze_dim_tecnico():
    return adicionar_auditoria(
        ler_postgres("dim_tecnico"),
        "dim_tecnico"
    )


@dp.materialized_view(
    name="bronze_dim_tempo"
)
def bronze_dim_tempo():
    return adicionar_auditoria(
        ler_postgres("dim_tempo"),
        "dim_tempo"
    )


@dp.materialized_view(
    name="bronze_dim_unidade"
)
def bronze_dim_unidade():
    return adicionar_auditoria(
        ler_postgres("dim_unidade"),
        "dim_unidade"
    )


@dp.materialized_view(
    name="bronze_fato_ocorrencia"
)
def bronze_fato_ocorrencia():
    return adicionar_auditoria(
        ler_postgres("fato_ocorrencia"),
        "fato_ocorrencia"
    )


@dp.materialized_view(
    name="bronze_fato_ordem_servico"
)
def bronze_fato_ordem_servico():
    return adicionar_auditoria(
        ler_postgres("fato_ordem_servico"),
        "fato_ordem_servico"
    )


@dp.materialized_view(
    name="bronze_fato_problema"
)
def bronze_fato_problema():
    return adicionar_auditoria(
        ler_postgres("fato_problema"),
        "fato_problema"
    )


@dp.materialized_view(
    name="bronze_tarefa"
)
def bronze_tarefa():
    return adicionar_auditoria(
        ler_postgres("tarefa"),
        "tarefa"
    )