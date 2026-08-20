"""Market knowledge bases for the Philippines and Indonesia (Q3).

Q1's knowledge base is Indian SME lending. The Q3 bots sell life insurance in
the Philippines and consumer financing in Indonesia. Pointing them at the
lending KB is exactly the "disconnected knowledge base" the assessment lists as
a rejection condition - and it showed immediately in testing: every Taglish
question fell below the retrieval threshold and the bot escalated three calls in
a row because it could not answer anything.

So each market gets its own knowledge base, tagged with `market`, and retrieval
is filtered by market. Same pipeline, same schema, same citation contract - only
the content differs.

These documents are SYNTHETIC and labelled as such, for the same reason as the
internal lending documents: the products are fictional, so no real insurer's or
lender's terms are being misrepresented. The domain content (grace periods,
lapse mechanics, denda, restructuring, payment channels) reflects how these
products genuinely work in each market, which is what makes the calls a real
test of localisation rather than of invented vocabulary.
"""
from __future__ import annotations

from pathlib import Path

from ...common.config import settings
from ...common.logging import log

BANNER = (
    "SYNTHETIC DOCUMENT - AUTHORED FOR A TECHNICAL ASSESSMENT. Fictional "
    "product of a fictional company. Not issued by, affiliated with, or based "
    "on any real company. Figures are illustrative."
)

# ---------------------------------------------------------------------------
# Philippines - life insurance / bancassurance
# ---------------------------------------------------------------------------
PH_KB = BANNER + """

# Kabuhayan Life - Product and Service Guide (Philippines)

## What products does Kabuhayan Life offer?
Three products: Kabuhayan Protect, a traditional term life plan; Kabuhayan
Secure, a whole life plan with a savings component; and Kabuhayan Bancassurance,
sold through partner bank branches to existing bank customers.

## What is the minimum and maximum coverage?
Coverage ranges from PHP 250,000 to PHP 5,000,000 for term life. Bancassurance
plans sold through bank branches range from PHP 500,000 to PHP 3,000,000.

## How much is the premium?
Premiums depend on age, coverage amount and payment frequency. For a 40-year-old
with PHP 1,000,000 coverage, the monthly premium starts at PHP 2,800. Premiums
may be paid monthly, quarterly, semi-annually or annually. Annual payment carries
a 5 percent discount.

## What is the grace period if I miss a premium?
There is a 31-day grace period from the due date. The policy stays in force
during the grace period and a claim within that window is still valid, with the
unpaid premium deducted from the benefit.

## What happens if my policy lapses?
If the premium is unpaid after the 31-day grace period, the policy lapses and
coverage stops. A lapsed policy may be reinstated within two years of the lapse
date, subject to payment of all overdue premiums plus interest, and a new
declaration of health. Reinstatement is not automatic.

## How do I pay my premium?
Payment channels: auto-debit from a bank account, over-the-counter at any
partner bank branch, GCash, Maya, or online bank transfer. Kabuhayan Life never
collects payment, card details or OTP over a phone call.

## Can I change my beneficiary?
Yes. A beneficiary change requires a signed change form and a valid ID. If the
beneficiary was designated as irrevocable, the existing beneficiary must consent
in writing.

## What is a rider?
A rider is an add-on benefit attached to the base policy. Available riders:
accidental death benefit, critical illness cover, waiver of premium on
disability, and hospital income benefit. Riders are priced separately and can be
added at policy anniversary.

## What is bancassurance and how is it different?
Bancassurance is insurance distributed through a partner bank. The bank refers
the customer, the policy is underwritten and serviced by Kabuhayan Life, and the
customer may service the policy either at the bank branch or directly.

## Is there a free-look period?
Yes, 15 days from receipt of the policy contract. Within that period the policy
may be returned for a refund of premiums paid, less any medical examination cost.

## What if I cannot afford the premium right now?
Options, in order: pay within the 31-day grace period; reduce the coverage
amount to lower the premium; change payment frequency; or, on a whole life
policy with accumulated cash value, apply an automatic premium loan. An agent
cannot waive a premium.

## OBJ-PH-01 "Is this a scam?"
Acknowledge that scam calls are common and do not argue. Confirm that no OTP,
card details or payment will ever be requested on the call. Offer to verify
identity by stating the last three digits of the policy number and the branch
where the policy was taken. Invite the customer to call the official hotline
back if they prefer.

## OBJ-PH-02 "Wala akong pambayad ngayon"
Acknowledge without judgement. Explain the 31-day grace period and confirm the
exact last date. Offer the documented alternatives: reduced coverage, changed
payment frequency, or a callback from the servicing officer. Never promise a
waiver or an extension beyond the grace period.

## OBJ-PH-03 "Masyadong mahal ang premium"
Do not argue about price. Establish what the coverage is protecting, then
explain that premium scales with coverage and that reducing coverage or changing
frequency lowers the payment. Mention the annual-payment discount.

## OBJ-PH-04 "I want to cancel my policy"
Do not process a cancellation on the call. Establish the reason. If within the
free-look period, explain the refund. Otherwise explain what is lost - coverage,
and any cash value - and offer a callback with the servicing officer before any
cancellation is actioned.

## Complaints and escalation
Level 1: the servicing branch or bank partner, response within 7 days. Level 2:
the customer service head, response within 15 days. Level 3: the Insurance
Commission. A customer who asks for a human must be transferred or scheduled
with a human officer on the same call.
"""

# ---------------------------------------------------------------------------
# Indonesia - multifinance / consumer financing
# ---------------------------------------------------------------------------
ID_KB = BANNER + """

# Sejahtera Multifinance - Panduan Produk dan Layanan (Indonesia)

## Produk apa saja yang tersedia?
Tiga produk: pembiayaan kendaraan bermotor (motor dan mobil), pembiayaan alat
usaha dan alat pertanian, serta pembiayaan multiguna dengan jaminan BPKB.

## Berapa DP dan tenor yang tersedia?
DP mulai dari 15 persen untuk motor dan 20 persen untuk mobil. Tenor tersedia
12, 18, 24, 36 dan 48 bulan. Tenor lebih panjang berarti angsuran lebih ringan
tetapi total biaya lebih besar.

## Kapan jatuh tempo angsuran?
Tanggal jatuh tempo ditetapkan saat kontrak dibuat, umumnya tanggal 5, 10, 15
atau 25 setiap bulan. Pembayaran sebaiknya dilakukan paling lambat satu hari
sebelum jatuh tempo agar dana terkonfirmasi.

## Berapa denda keterlambatan?
Denda keterlambatan sebesar 0,5 persen per hari dari nilai angsuran yang
tertunggak, dihitung sejak hari pertama setelah jatuh tempo. Denda maksimal
tidak melebihi nilai satu angsuran.

## Bagaimana cara membayar?
Kanal pembayaran: transfer ke virtual account, auto-debit rekening, Indomaret,
Alfamart, mobile banking, dan kantor cabang. Petugas tidak pernah meminta OTP,
PIN, atau menerima pembayaran tunai melalui telepon.

## Pembayaran saya sudah ditransfer tapi belum masuk sistem, bagaimana?
Konfirmasi tanggal dan kanal pembayaran, lalu minta bukti transfer dikirim ke
kanal resmi. Pembayaran melalui minimarket dan transfer antarbank dapat
memerlukan waktu satu hingga dua hari kerja untuk terkonfirmasi. Selama bukti
transfer valid, denda atas keterlambatan sistem akan ditinjau.

## Apa itu keringanan dan siapa yang bisa mengajukan?
Keringanan adalah penyesuaian pembayaran untuk nasabah yang mengalami kesulitan
sementara. Bentuknya: penjadwalan ulang tanggal dalam bulan yang sama,
pembayaran sebagian, atau restrukturisasi tenor. Pengajuan dinilai oleh tim
terkait berdasarkan riwayat pembayaran. Petugas telepon tidak berwenang
menyetujui keringanan.

## Bisakah saya melunasi lebih awal?
Bisa. Pelunasan dipercepat dikenakan biaya 3 persen dari sisa pokok. Nasabah
harus mengajukan permintaan perhitungan pelunasan terlebih dahulu.

## Bagaimana ketentuan penagihan?
Penagihan hanya dilakukan pukul 08.00 sampai 19.00 waktu setempat. Petugas
dilarang menggunakan bahasa yang mengancam, dilarang menghubungi pihak ketiga
sebelum tunggakan melewati 30 hari, dan wajib menawarkan opsi keringanan yang
tersedia sebelum eskalasi.

## Kapan kendaraan bisa ditarik?
Penarikan hanya dapat dilakukan setelah proses somasi resmi dan sesuai ketentuan
yang berlaku. Petugas telepon tidak berwenang menyatakan atau mengancam
penarikan kendaraan.

## OBJ-ID-01 "Belum ada uang bulan ini"
Akui kondisinya tanpa menghakimi. Jelaskan denda harian agar nasabah memahami
konsekuensinya, lalu tawarkan opsi keringanan yang terdokumentasi. Jangan
menjanjikan penghapusan denda.

## OBJ-ID-02 "Nanti saya kabari" (penolakan tidak langsung)
Ini bukan janji bayar. Minta tanggal yang spesifik dengan sopan. Jika nasabah
tidak dapat memastikan tanggal, catat sebagai belum ada komitmen, bukan janji
bayar, dan jadwalkan follow-up.

## OBJ-ID-03 "Kenapa dendanya besar sekali?"
Jelaskan perhitungan denda secara transparan: 0,5 persen per hari dari angsuran
tertunggak, dengan batas maksimal satu angsuran. Jika ada indikasi kesalahan
sistem atau pembayaran yang belum terkonfirmasi, ajukan peninjauan.

## OBJ-ID-04 "Saya mau bicara dengan petugas"
Hentikan proses dan alihkan. Konfirmasi nomor dan waktu yang tepat untuk
dihubungi. Permintaan ini tidak boleh dibantah atau ditunda.

## Pengaduan dan eskalasi
Tingkat 1: cabang atau customer service, tanggapan dalam 7 hari kerja. Tingkat
2: kepala layanan nasabah, tanggapan dalam 14 hari kerja. Tingkat 3: otoritas
terkait. Nasabah yang meminta berbicara dengan manusia harus dialihkan atau
dijadwalkan pada panggilan yang sama.
"""


def seed() -> dict[str, Path]:
    base = settings.raw_dir / "markets"
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    p = base / "ph_life_insurance_kb.md"
    p.write_text(PH_KB, encoding="utf-8")
    written["ph_kb"] = p

    p = base / "id_multifinance_kb.md"
    p.write_text(ID_KB, encoding="utf-8")
    written["id_kb"] = p

    for name, path in written.items():
        log("seed.market_written", doc=name, path=str(path), bytes=path.stat().st_size)
    return written


if __name__ == "__main__":
    seed()
