-- Minimal tables for single-user forward simulation.
-- Run in Supabase SQL editor before calling /api/sim/run-daily.

create table if not exists public.sim_orders (
  id bigserial primary key,
  report_id bigint not null,
  order_date date not null,
  ticker text not null,
  side text not null default 'BUY',
  order_amount numeric(18, 4) not null,
  signal_score numeric(18, 6),
  audit_status text not null default 'approved',
  audit_reason text,
  created_at timestamptz not null default now()
);

create index if not exists idx_sim_orders_ticker_date
  on public.sim_orders (ticker, created_at desc);

create table if not exists public.sim_order_audits (
  id bigserial primary key,
  report_id bigint not null,
  order_date date not null,
  ticker text not null,
  side text not null,
  order_amount numeric(18, 4) not null,
  signal_score numeric(18, 6),
  audit_status text not null,
  audit_reason text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_sim_order_audits_ticker_date
  on public.sim_order_audits (ticker, created_at desc);

