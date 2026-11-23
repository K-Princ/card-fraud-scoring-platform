from sqlalchemy import (
    BigInteger, String, Float, Numeric,
    DateTime, ForeignKey, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    transaction_amt: Mapped[float] = mapped_column(
        Numeric(12, 2), nullable=False
    )
    dist2: Mapped[float | None] = mapped_column(Float)
    d7: Mapped[float | None] = mapped_column(Float)
    d13: Mapped[float | None] = mapped_column(Float)
    product_cd: Mapped[str] = mapped_column(String(10), nullable=False)
    card4: Mapped[str | None] = mapped_column(String(30))
    card6: Mapped[str | None] = mapped_column(String(30))
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.transaction_id"), nullable=False
    )
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(50))
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )