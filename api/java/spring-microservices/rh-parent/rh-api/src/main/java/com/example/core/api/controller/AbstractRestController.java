package com.example.core.api.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.example.core.api.handler.IResponseHandler;

@Component
public abstract class AbstractRestController {
    @Autowired
    protected IResponseHandler responseHandler;
}
