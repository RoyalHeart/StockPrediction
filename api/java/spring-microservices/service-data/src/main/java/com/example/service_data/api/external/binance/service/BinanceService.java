package com.example.service_data.api.external.binance.service;

import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.binance.connector.client.SpotClient;
import com.binance.connector.client.impl.SpotClientImpl;
import com.binance.connector.client.utils.signaturegenerator.HmacSignatureGenerator;
import com.binance.connector.client.utils.signaturegenerator.SignatureGenerator;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
public class BinanceService {
    @Value("${binance.api_key}")
    private String API_KEY;
    @Value("${binance.api_secret}")
    private String API_SECRET;

    private final String BASE_URL = "https://api.binance.com";

    public Long getBalance() {
        SignatureGenerator signGenerator = new HmacSignatureGenerator(API_SECRET);
        SpotClient client = new SpotClientImpl(API_KEY, signGenerator, BASE_URL);
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("symbol", "TRUMPUSDT");
        String earn = client.createSimpleEarn().collateralRecord(parameters);
        String trade = client.createTrade().myTrades(parameters);
        log.info(">>> EARN {}", earn);
        log.info(">>> TRADE" + trade);
        return 1L;
    }
}
