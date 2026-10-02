# lct_logger.py
import csv
import math
import time


def classify_aoi(gaze_x, gaze_y, y_threshold=750):
  """Klassifiziert den Fokus anhand der Y-Bildschirmkoordinaten.

  - Y > y_threshold: HEADUNIT (unterer Bildschirmbereich)
  - sonst: ROAD
  """
  if gaze_x is None or gaze_y is None or gaze_x < 0 or gaze_y < 0:
    return 'UNKNOWN'

  if gaze_y > y_threshold:
    return 'HEADUNIT'
  else:
    return 'ROAD'


class LCTLogger:

  def __init__(
      self,
      start_x=0.0,
      finish_x=3000.0,
      gaze_estimator=None,
      aoi_threshold_y=750,
  ):
    self.start_x = start_x
    self.finish_x = finish_x
    self.gaze_estimator = gaze_estimator
    self.aoi_threshold_y = aoi_threshold_y

    self.is_recording = False
    self.completed = False

    self.start_time = None
    self.total_time = 0.0

    self.csv_file = None
    self.csv_writer = None

  def update(self, world, signs_data, webcam_frame=None):
    if world.player is None or self.completed:
      return

    p_trans = world.player.get_transform()
    p_vel = world.player.get_velocity()
    p_ctrl = world.player.get_control()

    cur_x = p_trans.location.x
    raw_y = p_trans.location.y
    y_norm = raw_y + 1.5  # Normalisiertes Y (0.0m bis 10.5m)
    speed_kmh = 3.6 * math.sqrt(p_vel.x**2 + p_vel.y**2 + p_vel.z**2)
    current_time = time.time()

    # --- 1. STARTLINIE BERÜHRT (X >= 0.0) ---
    if not self.is_recording and cur_x >= self.start_x:
      self.is_recording = True
      self.start_time = current_time

      filename = f'LCT_Log_{int(self.start_time)}.csv'
      self.csv_file = open(filename, mode='w', newline='')
      self.csv_writer = csv.writer(self.csv_file)

      # CSV Header inklusive Gaze- und AOI-Feldern
      self.csv_writer.writerow([
          'timestamp_sim',
          'elapsed_time_s',
          'pos_x',
          'pos_y_raw',
          'pos_y_norm',
          'speed_kmh',
          'steer',
          'active_sign_idx',
          'active_maneuver',
          'dist_to_next_sign',
          'gaze_x',
          'gaze_y',
          'aoi_focus',
          'line_type',
      ])

      world.hud.notification(
          '=== STARTLINIE ÜBERFAHREN: LOGGING GESTARTET ==='
      )
      print(
          f'[LCT LOG] Startlinie bei X={cur_x:.2f}m überfahren. Messung'
          ' gestartet!'
      )

    # --- 2. ZIELLINIE BERÜHRT (X >= 3000.0) ---
    elif self.is_recording and cur_x >= self.finish_x:
      self.is_recording = False
      self.completed = True
      self.total_time = current_time - self.start_time

      self.close()

      world.hud.notification(
          f'=== ZIEL ERREICHT! Zeit: {self.total_time:.2f}s ==='
      )
      print(
          f'[LCT LOG] Ziellinie bei X={cur_x:.2f}m erreicht!'
          f' Gesamtzeit: {self.total_time:.2f}s'
      )

    # --- 3. DATEN SCHREIBEN (Während Messung aktiv ist) ---
    if self.is_recording and self.csv_writer:
      active_idx = -1
      active_maneuver = 'None'
      dist_to_next = 999.0

      for idx, sign in enumerate(signs_data):
        if isinstance(sign, dict):
            sign_x = sign.get('x_pos', sign.get('x', 0.0))
            maneuver = sign.get('maneuver', 'Unknown')
        else:
            _, sign_x, maneuver = sign

      # Linientyp anhand der empirischen Y-Grenzen
      line_type = 'Solid' if (raw_y <= -7.3 or raw_y >= 3.8) else 'Broken'
      elapsed_s = current_time - self.start_time

      # EyeTrax Gaze & AOI Bestimmung
      gaze_x, gaze_y = -1.0, -1.0
      aoi_focus = 'UNKNOWN'

      if self.gaze_estimator and webcam_frame is not None:
        try:
          gaze_x, gaze_y = self.gaze_estimator.predict(webcam_frame)
          aoi_focus = classify_aoi(gaze_x, gaze_y, self.aoi_threshold_y)
        except Exception:
          pass

      self.csv_writer.writerow([
          round(world.hud.simulation_time, 3),
          round(elapsed_s, 3),
          round(cur_x, 3),
          round(raw_y, 3),
          round(y_norm, 3),
          round(speed_kmh, 2),
          round(p_ctrl.steer, 4),
          active_idx,
          active_maneuver,
          round(dist_to_next, 2),
          round(gaze_x, 1) if gaze_x != -1.0 else -1,
          round(gaze_y, 1) if gaze_y != -1.0 else -1,
          aoi_focus,
          line_type,
      ])

  def close(self):
    if self.csv_file is not None:
      self.csv_file.close()
      self.csv_file = None
      print('[LCT LOG] CSV-Datei gespeichert und geschlossen.')
