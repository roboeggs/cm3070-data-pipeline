import datetime

class SupabaseStorage:
    def __init__(self, supabase):
        self.supabase = supabase

    def get_processing_log(self):
        resp = self.supabase.table('processing_log').select('ticker, last_processed_time').execute()
        return {row['ticker']: row['last_processed_time'] for row in resp.data}

    def save_processing_log(self, ticker, first_time, last_time, params_json):
        self.supabase.table('processing_log').upsert({
            'ticker': ticker,
            'first_loaded_time': first_time,
            'last_processed_time': last_time,
            'params_json': params_json
        }, on_conflict='ticker').execute()

    def save_detector_state(self, ticker, state_json):
        has_active_levels = bool(
            state_json.get('confirmed_support_levels')
            or state_json.get('confirmed_resistance_levels')
        )
        self.supabase.table('detector_state').upsert({
            'ticker': ticker,
            'state_blob': state_json,
            'has_active_levels': has_active_levels,
        }, on_conflict='ticker').execute()

    def load_detector_state(self, ticker):
        resp = self.supabase.table('detector_state').select('*').eq('ticker', ticker).execute()
        return resp.data[0] if resp.data else None


    def get_active_detector_states(self):
        """Returns all detector_state rows where has_active_levels = True."""
        resp = (
            self.supabase
            .table('detector_state')
            .select('ticker, saved_at, state_blob')
            .eq('has_active_levels', True)
            .execute()
        )
        return resp.data

    def build_active_levels_summary(self):
        """Collects a summary of all active levels in JSON format."""
        processing_log = self.get_processing_log()
        active_states = self.get_active_detector_states()
        tickers_list = []

        for state in active_states:
            ticker = state['ticker']
            blob = state['state_blob']
            support_levels = blob.get('confirmed_support_levels', [])
            resistance_levels = blob.get('confirmed_resistance_levels', [])
            all_levels = support_levels + resistance_levels

            if not all_levels:
                continue  # skip tickers without active levels

            levels_list = []
            current_price = None
            quote_volume_24h = None

            for idx, level in enumerate(all_levels):
                samples = level.get('samples', [])
                last_sample = samples[-1] if samples else {}

                # current_price and quote_volume_24h are taken from the first level (the last sample)
                if idx == 0 and last_sample:
                    current_price = last_sample.get('center')
                    quote_volume_24h = last_sample.get('quote_volume_ratio_24h')

                # Main level fields
                level_id = level.get('level_id')
                level_type = level.get('level_type')
                center = level.get('center')
                zone_low = level.get('zone_low')
                zone_high = level.get('zone_high')
                touch_count = level.get('touch_count')
                first_touch_time = level.get('first_touch_time')
                last_touch_time = level.get('last_touch_time')

                # Fields from the last sample
                recent_touch_count = last_sample.get('recent_touch_count')
                is_confirmation_touch = last_sample.get('is_confirmation_touch')
                level_age_bars = last_sample.get('level_age_bars')
                distance_to_level_pct = last_sample.get('distance_to_level_pct')
                zone_width_pct = last_sample.get('zone_width_pct')

                # ML forecast
                ml_prediction = level.get('ml_prediction')
                if ml_prediction:
                    breakout_1h = ml_prediction.get('breakout_1h', {}).get('probability', 0.0)
                    breakout_4h = ml_prediction.get('breakout_4h', {}).get('probability', 0.0)
                    breakout_24h = ml_prediction.get('breakout_24h', {}).get('probability', 0.0)
                    recommended = ml_prediction.get('recommended_horizon')
                    is_candidate = ml_prediction.get('is_candidate', False)
                else:
                    breakout_1h = 0.0
                    breakout_4h = 0.0
                    breakout_24h = 0.0
                    recommended = None
                    is_candidate = False

                level_dict = {
                    "id": level_id,
                    "type": level_type,
                    "center": center,
                    "zone_low": zone_low,
                    "zone_high": zone_high,
                    "zone_width_pct": zone_width_pct,
                    "touch_count": touch_count,
                    "first_touch_time": first_touch_time,
                    "last_touch_time": last_touch_time,
                    "recent_touch_count": recent_touch_count,
                    "is_confirmation_touch": is_confirmation_touch,
                    "level_age_bars": level_age_bars,
                    "distance_to_level_pct": distance_to_level_pct,
                    "ml": {
                        "breakout_1h": breakout_1h,
                        "breakout_4h": breakout_4h,
                        "breakout_24h": breakout_24h,
                        "recommended": recommended,
                        "is_candidate": is_candidate
                    }
                }
                levels_list.append(level_dict)

            updated_at = processing_log.get(ticker)

            ticker_dict = {
                "symbol": ticker,
                "current_price": current_price,
                "quote_volume_24h": quote_volume_24h,
                "updated_at": updated_at,
                "levels": levels_list
            }
            tickers_list.append(ticker_dict)

        summary = {
            "schema_version": 1,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "tickers": tickers_list
        }
        return summary