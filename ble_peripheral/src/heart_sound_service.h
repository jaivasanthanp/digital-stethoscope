#ifndef HEART_SOUND_SERVICE_H
#define HEART_SOUND_SERVICE_H

#include <stdint.h>
#include <zephyr/bluetooth/conn.h>

void hsc_service_init(void);

int hsc_service_notify(struct bt_conn *conn,
                       uint8_t class_id, uint8_t confidence,
                       uint32_t timestamp_ms);

#endif /* HEART_SOUND_SERVICE_H */
