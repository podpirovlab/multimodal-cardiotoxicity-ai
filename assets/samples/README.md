# Sample recording

`twadb-twa36.edf` is record **twa36** of the PhysioNet/Computing in Cardiology Challenge 2008
T-Wave Alternans Database (100 two-minute records), itself taken from the PTB Diagnostic ECG
Database (`ptbdb/patient093/s0396lre`). It is the first 109 seconds (whole EDF data records) of a
standard 12-lead ECG at 500 Hz, converted from WFDB to EDF+ so that the web tool can open it with
one click. The original stores 0.5 µV per step; the EDF file uses a finer 16-bit step of 0.052 µV,
so no sample differs from the original by more than 0.052 µV.

It belongs to the half of the challenge records used for developing the algorithm, not to the
held-out half (README §7.1).

Licence: Open Data Commons Attribution License v1.0. Please cite:
- Moody GB. The PhysioNet/Computers in Cardiology Challenge 2008: T-wave alternans.
  *Computers in Cardiology* 2008;35:505–508.
- Bousseljot R, Kreiseler D, Schnabel A. Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über
  das Internet. *Biomedizinische Technik* 1995;40(S1):317.
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation* 2000;101(23):e215–e220.
